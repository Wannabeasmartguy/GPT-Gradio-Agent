from langchain_community.chat_models import ChatOllama
from langchain_openai.chat_models.azure import AzureChatOpenAI
from langchain_core.runnables.base import RunnableSerializable
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser,StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.tools.render import render_text_description

from operator import itemgetter
from typing import List,Literal
from dotenv import load_dotenv
import os
load_dotenv()


def create_llm(model_type:Literal["OpenAI","Ollama"],
               model_choice:str,
               temperature:float=0.7):
    '''根据传入的模型类型，选择对应的模型'''
    if model_type == 'OpenAI':
        llm = AzureChatOpenAI(model=model_choice,
                        openai_api_type=os.getenv('API_TYPE'),
                        azure_endpoint=os.getenv('AZURE_OAI_ENDPOINT'),
                        openai_api_key=os.getenv('AZURE_OAI_KEY'),
                        openai_api_version=os.getenv('API_VERSION'),
                        # eployment_name=os.getenv('AZURE_OAI_ENDPOINT')+ "deployments/" +'gpt-35-turbo', 
                        deployment_name=model_choice,
                        temperature=temperature)
    elif model_type == 'Ollama':
        llm = ChatOllama(model=model_choice,
                         temperature=temperature)
    
    return llm


class CommonAgent:
    def __init__(self, llm,tools:list, agent_kwargs={}):
        self.llm = llm
        self.tools = tools
        self.agent_kwargs = agent_kwargs
        
        self.tools_choose_prompt = self.init_tools_choose_prompt()
        self.main_sys_prompt = self.init_main_sys_prompt()
        self.chain = self.init_chain()
    
    def init_tools_chain(self,model_output):
        '''
        将工具初始化为一个 chain 

        Args:
            model_output: 模型输出的工具信息,
        
        Returns:
            LCEL——工具的输入参数 | 对应选中的工具
        '''

        tool_map = {tool.name: tool for tool in self.tools}
        chosen_tool = tool_map[model_output["name"]]
        return itemgetter("arguments") | chosen_tool
    
    def init_tools_choose_prompt(self) -> ChatPromptTemplate:
        '''Initialize the prompt for the tools choose'''

        rendered_tools = render_text_description(self.tools)

        system_prompt = f"""You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

        {rendered_tools}

        Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys.

        Requirements: NO opinions, suggestions, opinions and requests shall be made, and THE ABOVE CONTENT shall be STRICTLY OBSERVER.
        """

        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", "{input}")]
        )
        return prompt
    
    def init_main_sys_prompt(self) -> ChatPromptTemplate:
        '''Initialize the prompt for the main chat'''

        main_sys_prompt = f"""你是一个 AI 助手，以下是你可能用得上的内容

        {'{text}'}

        请你根据以上内容，回答用户的问题。
        """

        prompt = ChatPromptTemplate.from_messages(
            [("system", main_sys_prompt),("user", "{input}")]
        )

        return prompt
    
    def init_chain(self) -> RunnableSerializable:
        '''Initialize the complete chain'''
        
        chain = (
            self.tools_choose_prompt 
            | self.llm
            | JsonOutputParser() 
            | self.init_tools_chain
            | { "text":StrOutputParser(),"input":lambda query: query} 
            | self.main_sys_prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
    
    def invoke(self,input:dict) -> str:
        '''Transform the input into the llm, and then get the output'''
        return self.chain.invoke(input)
    

class OpenAIChatAgent:
    def __init__(self, llm, tools:list):
        self.llm = llm.bind_tools(tools)
        self.tools = tools
        self.MEMORY_KEY = "chat_memory"
        self.chat_memory = ConversationBufferMemory(memory_key=self.MEMORY_KEY, return_messages=True)
        # self.chat_memory = memory

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are very powerful assistant, but bad at calculating lengths of words.",
                ),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        
        self.agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                self.MEMORY_KEY: lambda x: x[self.MEMORY_KEY],
            }
            | self.prompt
            | self.llm
            | OpenAIToolsAgentOutputParser()
        )
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def invoke(self, query:str) -> str:
        '''
        Transform the input into the llm, and then get the output
        
        Args:
            query: 用户输入的问题
        '''
        chat_history = self.chat_memory.load_memory_variables({})[self.MEMORY_KEY]
        answer = self.agent_executor.invoke({"input": query,self.MEMORY_KEY: chat_history})
        # self.chat_memory.save_context({"input": query},{"output": answer["output"]})
        return answer["output"]
    
    def get_memory(self) -> List[BaseMessage]:
        '''Get the memory of the agent'''
        return self.chat_memory.load_memory_variables({})[self.MEMORY_KEY]
    
    def get_memory_list(self) -> List[str]:
        '''Get the memory of the agent(pure text)'''
        return [message.content for message in self.get_memory()]
