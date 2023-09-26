import gradio as gr
import openai
import time
import os
from dotenv import load_dotenv
import pandas
# from utils import *

# import langchain to chat with file
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import DirectoryLoader,PyPDFLoader,UnstructuredFileLoader
from langchain.chains import RetrievalQA

load_dotenv()

openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv('OPENAI_API_KEY')
openai.api_version = os.getenv('OPENAI_API_VERSION')
openai.api_type = os.getenv('OPENAI_API_TYPE')

# initialize the embedding model setting 
embedding_model = "text-embedding-ada-002"

#gr.Chatbot.postprocess = format_io

# <---------- set environmental parameters --------->

def deliver(message:str,
            model_choice:str,
            chat_history:list, 
            chat_history_list:list,
            system:str,
            context_length:int, 
            temperature:float,
            max_tokens:int,
            top_p:float,
            frequency_penalty:float,
            presence_penalty:float):
    '''
    Response function for chat-only
    '''

    # System Prompt and User Prompt
    if system:
        system_input = {
            "role": "system",
            "content": system
        }
        if chat_history == []:
            chat_history.append(system_input)
        else:
            chat_history[0] = system_input
 
    user_input = {
        "role": "user",
        "content": message
    }

    chat_history.append(user_input)

    if context_length == 0:
        # If context_length == 0,clean up chat_history
        response = openai.ChatCompletion.create(
            engine=model_choice,
            messages=[system_input,user_input],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=None
        )
    else:
        response = openai.ChatCompletion.create(
            engine=model_choice,
            messages=chat_history,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=None
        )
    reply = response.choices[0].message.content
        
    # GPT reply
    chat_input = {
        "role": "assistant",
        "content": reply
    }
    chat_history.append(chat_input)
    chat_history_list.append([message,None])

    # Trim the context length first
    if (len(chat_history)-1 > context_length) and len(chat_history)>3:
        chat_history = [chat_history[0]]+chat_history[-context_length:]
    
    return chat_history_list,message,chat_history

def stream(history_list:list,chat_history:list[dict]):
    '''
    Used to make LLM output looks like stream(Not real stream output).
    '''
    bot_message = chat_history[-1]['content']
    history_list[-1][1] = ""
    for character in bot_message:
        history_list[-1][1] += character
        time.sleep(0.02)
        yield history_list

def upload_file(file_obj,
                file_ask_history_list:list,
                question_prompt: str,
                file_answer:list
                ):
    '''
    Upload your file to chat \n
      \n
    return: 
    file_ask_history_list:list[list]
    result:dict
    '''

    # load your document
    loader = UnstructuredFileLoader(file_obj.name)
    document = loader.load()

    # initialize splitter
    text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=10)
    split_docs = text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings()
    llm = AzureChatOpenAI(model="gpt-35-turbo",
                    openai_api_type="azure",
                    deployment_name="gpt-35-turbo", # <----------设置选择模型的时候修改这里
                    temperature=0.7)
    
    docsearch = Chroma.from_documents(split_docs, embeddings)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                                        retriever=docsearch.as_retriever(), 
                                        return_source_documents=True)
    result = qa({"query": question_prompt})
    usr_prob = result["query"]
    #ai_answer = result["result"]
    file_answer[0] = result
    file_ask_history_list.append([usr_prob,None])
    return file_ask_history_list,file_answer

def file_ask_stream(file_ask_history_list:list[list],file_answer:list):
    '''
    Used to make file-answer looks like stream;\n
    'file_ask_history_list' will be transfered to chatbot
    '''
    bot_message = file_answer[0]["result"]
    file_ask_history_list[-1][1] = ""
    for character in bot_message:
        file_ask_history_list[-1][1] += character
        time.sleep(0.02)
        yield file_ask_history_list

def rst_mem(chat_his:list):
    '''
    Reset the chatbot memory(chat_his).
    '''
    chat_his = []
    return chat_his

# <---------- GUI ---------->
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        '''
        # <center>GPT AGENT<center>
        <center>Use the agent make your work and life much more efficient.<center>
        '''
    )
    usr_msg = gr.State()
    chat_his = gr.State([])
    with gr.Row():
        with gr.Column(scale=1.8):
            model_choice = gr.Radio(choices=["gpt-35-turbo","gpt-35-turbo-16k","gpt-4"],
                                    value="gpt-35-turbo",
                                    label="Model",info="支持模型选择，立即生效")
            chat_bot = gr.Chatbot(height=500,
                                show_copy_button=True,
                                bubble_full_width=False)
            message = gr.Textbox(label="Input your prompt",
                                        info="'Shift + Enter' to begin an new line. Press 'Enter' can also send your Prompt to the LLM.")
            with gr.Row(scale=0.1):
                clear = gr.ClearButton([message, chat_bot,chat_his],scale=1,size="sm")
                send = gr.Button("Send",scale=2)

        with gr.Column():
            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion("Commom Setting"):
                            System_Prompt = gr.Textbox("You are a helpful AI.", label="System Prompt",
                                                    info="'Shift + Enter' to begin an new line.")
                            Context_length = gr.Slider(0, 32, value=4, step=1, label="Context length",
                                                    info="每次请求携带的历史消息数")                    
            
                        with gr.Accordion("Additional Setting"):
                            max_tokens = gr.Slider(0, 4096, value=400, step=1, label="max_tokens",
                                                info="携带上下文交互的最大 token 数")
                            Temperature = gr.Slider(0, 2, value=0.5, step=0.1, label="Temperature",
                                                    info="随机性：值越大，回复越随机")
                            top_p = gr.Slider(0, 1, value=1, step=0.1, label="top_p",
                                            info="核采样：与随机性类似，但不要与随机性一起修改")
                            frequency_penalty = gr.Slider(-2, 2, value=0, step=0.1, label="frequency_penalty",
                                                        info="频率惩罚度：值越大，越不容易出现重复字词")
                            presence_penalty = gr.Slider(-2, 2, value=0, step=0.1, label="frequency_penalty",
                                                        info="话题新鲜度：值越大，越可能扩展到新的话题")
            with gr.Tab("chatfiles"):
                # set a element to aviod indexerror
                file_answer = gr.State(['0']) 
                file = gr.File(label="The file you want to chat with")
                chat_with_file = gr.Button(value="Chat with file")

    # Merge all handles that require input and output.
    input_param = [message, model_choice, chat_his, chat_bot, System_Prompt, 
                   Context_length, Temperature,max_tokens,top_p,frequency_penalty,
                   presence_penalty]
    output_param = [chat_bot, usr_msg, chat_his]

    # chatbot button event
    message.submit(deliver,input_param, output_param, queue=False).then(stream,[chat_bot,chat_his],chat_bot)
    send.click(deliver,input_param, output_param, queue=False).then(stream,[chat_bot,chat_his],chat_bot)
    clear.click(rst_mem,inputs=chat_his,outputs=chat_his)
    # chat_file button event
    chat_with_file.click(upload_file,inputs=[file,chat_bot,message,file_answer],outputs=[chat_bot,file_answer]).then(file_ask_stream,[chat_bot,file_answer],[chat_bot])

demo.queue().launch(inbrowser=False,debug=True,
                    #auth=[("admin","123456")],auth_message="欢迎使用 GPT-Gradio-Agent ,请输入用户名和密码"
                    )