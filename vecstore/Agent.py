from langchain.prompts import PromptTemplate
from langchain.chains import LLMRequestsChain, LLMChain
from langchain_community.chat_models import AzureChatOpenAI
from openai import BadRequestError
import gradio as gr
from pydantic.error_wrappers import ValidationError

def url_request_chain(model:str,
              url:str,
              chat_bot:list[list],
              template:str):
    '''
    构建一个总结网页内容（文章）的 LLMRequestsChain ，非 Agent
    '''

    llm = AzureChatOpenAI(model=model,
                    openai_api_type="azure",
                    deployment_name=model, # <----------设置选择模型的时候修改这里
                    temperature=0)
    
    if url == "" or None:
        raise gr.Error(" 请输入 URL ")
    if template == "" or None:
        raise gr.Error(" 请选择 Prompt 模版，或自行设置 Prompt 模版 ")

    try:
        prompt = PromptTemplate(
            input_variables=["requests_result"],
            template=template
        )
    except ValidationError:
        raise gr.Error(" Prompt 模板格式存在问题，请修改 Prompt ")

    chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=prompt))

    inputs = {
    "url": url
    }

    try:
        response = chain(inputs)
        # print(response['output'])
        chat_bot.append([f"请为我总结这篇文章:{url}",response['output']])
        return chat_bot
    except BadRequestError:
        raise gr.Error("待总结的文章长度已经超过了模型能够承受的最大长度！请选择支持更长上下文的模型！")    