import gradio as gr
import openai
import time
import os
from dotenv import load_dotenv
import pandas
from vecstore import *

# import langchain to chat with file
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import AzureChatOpenAI
from langchain.document_loaders import DirectoryLoader,PyPDFLoader,UnstructuredFileLoader
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain

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
                split_tmp,
                ):
    '''
    Upload your file to chat \n
      \n
    return: 
    list of files are splitted.
    '''

    # load your document
    loader = UnstructuredFileLoader(file_obj.name)
    document = loader.load()

    # initialize splitter
    text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=10)
    split_docs = text_splitter.split_documents(document)
    split_tmp.append(split_docs)
    return split_tmp

def ask_file(split_docs:list,
            file_ask_history_list:list,
            question_prompt: str,
            file_answer:list,
            model_choice:str,
            sum_type:str):
    '''
    send splitted file to LLM
    '''
    embeddings = OpenAIEmbeddings()
    llm = AzureChatOpenAI(model=model_choice,
                    openai_api_type="azure",
                    deployment_name=model_choice, 
                    temperature=0.7)
    if split_docs[-1] != None:
        docsearch = Chroma.from_documents(split_docs[-1], embeddings)
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type=sum_type, 
                                            retriever=docsearch.as_retriever(), 
                                            return_source_documents=True)
        result = qa({"query": question_prompt})
        usr_prob = result["query"]
    # if there is no file, let it become a common chat model
    else:
        result = llm(question_prompt)
        usr_prob = question_prompt
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

def summarize_file(split_docs,chatbot,model_choice,sum_type):
    llm = AzureChatOpenAI(model=model_choice,
                    openai_api_type="azure",
                    deployment_name=model_choice, # <----------设置选择模型的时候修改这里
                    temperature=0.7)
    # 创建总结链
    chain = load_summarize_chain(llm, chain_type=sum_type, verbose=True)
    
    # 执行总结链
    summarize_result = chain.run(split_docs[-1])

    # 构造 chatbox 格式
    chatbot.append(["Please summarize the file for me.",None])
    return summarize_result,chatbot

def sum_stream(summarize_result,chatbot):
    '''
    Used to make summarized result be outputed as stream.
    '''
    chatbot[-1][1] = ""
    for character in summarize_result:
        chatbot[-1][1] += character
        time.sleep(0.02)
        yield chatbot

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
                split_tmp = gr.State(['0'])
                sum_result = gr.State()
                # set a element to aviod indexerror
                file_answer = gr.State(['0']) 
                
                with gr.Column():
                    file = gr.File(label="The file you want to chat with")
                    vector_path = gr.Text(label="Knowledge base save path",
                                          info="Choose the folder you want to save, and PASTE THE ABSOLUTE PATH here")
                    vector_content = gr.DataFrame(label="Knowledge Base Document Catalog",
                                                  interactive=False,
                                                  )
                    with gr.Column():
                        load_vec = gr.Button(value="Load your knowledge base")
                        add_file = gr.Button(value="Add file to knowledge base")
                    sum_type = gr.Radio(choices=[("小文件(file with few words)","stuff"),("大文件(file with a large word count)","refine")],
                                        value="stuff",
                                        label="Choose the type of file to be summarized",
                                        info="如果待总结字数较多，请选择“大文件”（选小文件可能导致超出 GPT 的最大 Token ）")
                    with gr.Row():
                        chat_with_file = gr.Button(value="Chat with file")
                        summarize = gr.Button(value="Summarize")

    # Merge all handles that require input and output.
    input_param = [message, model_choice, chat_his, chat_bot, System_Prompt, 
                   Context_length, Temperature,max_tokens,top_p,frequency_penalty,
                   presence_penalty]
    output_param = [chat_bot, usr_msg, chat_his]

    # chatbot button event
    message.submit(deliver,input_param, output_param, queue=False).then(stream,[chat_bot,chat_his],chat_bot)
    send.click(deliver,input_param, output_param, queue=False).then(stream,[chat_bot,chat_his],chat_bot)
    clear.click(rst_mem,inputs=chat_his,outputs=chat_his)

    message.submit(lambda: gr.update(value=''), [],[message])
    send.click(lambda: gr.update(value=''), [],[message])
    
    # chat_file button event
    file.upload(upload_file,inputs=[file,split_tmp],outputs=[split_tmp],show_progress="full")
    #file.clear(lambda: gr.update(value=None), [],[file])
    #file.clear(lambda split_tmp: gr.update(value=None), [],[split_tmp])
    chat_with_file.click(ask_file,inputs=[split_tmp,chat_bot,message,file_answer,model_choice,sum_type],outputs=[chat_bot,file_answer]).then(file_ask_stream,[chat_bot,file_answer],[chat_bot])
    summarize.click(summarize_file,inputs=[split_tmp,chat_bot,model_choice,sum_type],outputs=[sum_result,chat_bot]).then(sum_stream,[sum_result,chat_bot],[chat_bot])

    chat_with_file.click(lambda: gr.update(value=''), [],[message])
    summarize.click(lambda: gr.update(value=''), [],[message])

    vector_path.blur(create_vectorstore,inputs=[vector_path])
    load_vec.click(load_vectorstore,inputs=[vector_path],outputs=[vector_content])
    add_file.click(add_file_in_vectorstore,inputs=[vector_path,split_tmp])

demo.queue().launch(inbrowser=True,debug=True,
                    #auth=[("admin","123456")],auth_message="欢迎使用 GPT-Gradio-Agent ,请输入用户名和密码"
                    )