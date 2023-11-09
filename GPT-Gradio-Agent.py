import gradio as gr
import openai
import time
import os
from dotenv import load_dotenv
import pandas
from vecstore.vecstore import * 
from gga_utils.common import *
from gga_utils.theme import *

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

# 初始化主题
set_theme = adjust_theme()

#gr.Chatbot.postprocess = format_io

# <---------- set environmental parameters --------->

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
                progress=gr.Progress()
                ):
    '''
    Upload your file to chat \n
      \n
    return: 
    list of files are splitted.
    '''
    from pdf2image.exceptions import PDFInfoNotInstalledError
    try:
        # load your document
        loader = UnstructuredFileLoader(file_obj.name)
        document = loader.load()
        progress(0.3, desc="Loading the file...")
    except (FileNotFoundError,PDFInfoNotInstalledError):
        raise gr.Error("File upload failed. This may be due to formatting issues (non-standard formats)")

    # initialize splitter
    text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=10)
    split_docs = text_splitter.split_documents(document)
    split_tmp.append(split_docs)
    progress(1, desc="Dealing...")
    gr.Info("Processing completed.")

    return split_tmp,gr.File(label="The file you want to chat with")

def file_ask_stream(file_ask_history_list:list[list],file_answer:list):
    '''
    Used to make file-answer looks like stream;\n
    'file_ask_history_list' will be transfered to chatbot
    '''
    try:
        bot_message = file_answer[0]["answer"]
    except TypeError:
        raise gr.Error("No model response obtained")
    file_ask_history_list[-1][1] = ""
    for character in bot_message:
        file_ask_history_list[-1][1] += character
        time.sleep(0.02)
        yield file_ask_history_list

def sum_stream(summarize_result,chatbot):
    '''
    Used to make summarized result be outputed as stream.
    '''
    chatbot[-1][1] = ""
    for character in summarize_result:
        chatbot[-1][1] += character
        time.sleep(0.02)
        yield chatbot

# <---------- GUI ---------->
with gr.Blocks(theme=set_theme,css='style\style.css') as demo:
    gr.Markdown(
        '''
        # <center>GPT AGENT<center>
        <center>Use the agent make your work and life much more efficient.<center>
        <center>📁 means *knowledgebase* in the interface.<center>
        '''
    )
    usr_msg = gr.State()
    chat_his = gr.State([])
    # chat_memory = gr.State(ConversationBufferMemory(memory_key="chat_memory", return_messages=True))
    with gr.Row():
        with gr.Column(elem_id="history"):
            with gr.Row():
                delete_dialog = gr.Button(
                    icon=r"icon\delete_dialog.png",
                    value="Delete Dialog",
                    min_width=5,
                    elem_id="btn_transparent",
                    size="sm",
                )
                add_dialog = gr.ClearButton(
                    components=[chat_his],
                    icon=r"icon\add_dialog.png",
                    #variant="primary",
                    value="New Dialog",
                    min_width=5,
                    elem_id="btn_transparent",
                    size="sm"
                )
            His_choice_cache = get_all_conversation_names()
            Historylist = gr.Radio(
                #label="Dialog Box",
                show_label=False,
                interactive=True,
                value=list_vali_check(His_choice_cache),
                choices=His_choice_cache,
                elem_id="history-select-dropdown",
            )
        with gr.Column(scale=2):
            with gr.Group():
                model_choice = gr.Radio(choices=["gpt-35-turbo","gpt-35-turbo-16k","gpt-4"],
                                    value="gpt-35-turbo",
                                    label="Model",info="支持模型选择，立即生效")
                chat_name = gr.Textbox(label="Chatbot name",
                                       interactive=True,
                                       value=get_last_conversation_name(),
                                       info="对话名称将被用于导出聊天记录时的文件命名。")
            chat_bot = gr.Chatbot(height=500,
                                  value=get_last_conversation_content(),
                                  show_label=False,
                                  show_copy_button=True,
                                  bubble_full_width=False)
            with gr.Row():
                message = gr.Textbox(label="Input your prompt",
                                     info="'Shift + Enter' to begin an new line. Press 'Enter' can also send your Prompt to the LLM.",
                                     scale=7)
                export_his = gr.Button(value="Export Chat History",scale=1)
            with gr.Row():
                clear = gr.ClearButton([message, chat_bot,chat_his],scale=1,size="sm")
                send = gr.Button("Send",variant='primary',elem_id="btn",scale=2)
            with gr.Row():
                chat_with_file = gr.Button(value="Chat with file (Valid for 📁)")
                summarize = gr.Button(value="Summarize (Valid only for uploaded file)")

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
            with gr.Tab("RAG"):
                split_tmp = gr.State(['0'])
                sum_result = gr.State()
                # set a element to aviod indexerror
                file_answer = gr.State(['0']) 
                
                with gr.Column():
                    with gr.Group():
                        file = gr.File(label="The file you want to chat with")
                        with gr.Row():
                            estimate_cost = gr.Text(label="Estimated cost:", 
                                                    info="Estimated cost of embed file",
                                                    scale=2)
                            refresh_file_cost = gr.Button(value="Refresh file and estimate cost",
                                                          scale=1)

                    with gr.Group():
                        vector_path = gr.Text(label="Knowledge base save path",
                                            info="Choose the folder you want to save, and PASTE THE ABSOLUTE PATH here")
                        with gr.Row():
                            vector_content = gr.DataFrame(#label="Knowledge Base Document Catalog",
                                                          value = pd.DataFrame(columns=['文件名称']),
                                                          visible=False,
                                                          interactive=False,
                                                         )
                            file_list = gr.Dropdown(interactive=True,
                                                    # allow_custom_value=True,
                                                    label="File list")
                        with gr.Column():
                            create_vec_but = gr.Button(value="Create a new knowledge base 📁")
                            load_vec = gr.Button(value="Load your 📁 ",variant='primary',elem_id="btn")
                            with gr.Row():
                                add_file = gr.Button(value="Add it (The file uploaded) to 📁")
                                delete_file = gr.Button(value="Delete it (Selected in dropdown) from 📁")  
                    with gr.Accordion("File chat setting"):
                        filter_choice = gr.Radio(choices=["All", "Selected file"],
                                                value="All",
                                                label="Search scope",
                                                info="“All” means whole knowledge base;“Selected file” means the file selected in dropdown")
                        sum_type = gr.Radio(choices=[("small file","stuff"),
                                                     ("large file(refine)","refine"),
                                                     ("large file(map reduce)","map_reduce"),
                                                     ("large file(map rerank, for chat)","map_rerank")],
                                            value="refine",
                                            label="File size type",
                                            info="也作用于“Summarize”。如果待总结字数较多，请选择“lagre size”（选“small size”可能导致超出 GPT 的最大 Token ）")

    # Radio control
    add_dialog.click(add_conversation_to_json,
                     inputs=[chat_name,chat_bot]
                     ).success(lambda:gr.Radio(choices=get_all_conversation_names(),
                                                value=get_last_conversation_name()),
                                                outputs=Historylist
                                ).success(lambda: gr.Chatbot(value=''), 
                                          [],
                                          [chat_bot]
                                          ).success(lambda: gr.Textbox(value=get_last_conversation_name()), 
                                                    [],
                                                    [chat_name]
                                                    )
    
    delete_dialog.click(delete_conversation_from_json,
                        inputs=[chat_name]
                        ).success(lambda: gr.Radio(choices=get_all_conversation_names(), 
                                                   value=get_last_conversation_name()), 
                                                   outputs=[Historylist] 
                                  ).success(get_last_conversation_content,
                                            [],
                                            [chat_bot]).success(lambda Historylist:gr.Textbox(value=Historylist),
                                                                [Historylist],
                                                                [chat_name])
    
    Historylist.select(lambda:gr.Radio(),[],[Historylist]
                       ).success(lambda Historylist: gr.Textbox(value=Historylist),
                                 [Historylist],[chat_name]
                                 ).success(get_selected_conversation_content,
                                           [chat_name],
                                           [chat_bot])

    chat_name.blur(modify_conversation_name,
                   inputs=[Historylist,chat_name],
                   outputs=[chat_name]).success(lambda chat_name: gr.Radio(
                                                                    show_label=False,
                                                                    interactive=True,
                                                                    value=chat_name,
                                                                    choices=get_all_conversation_names(),
                                                                    ),
                                                            inputs=chat_name,outputs=[Historylist]
                                                )

    # Merge all handles that require input and output.
    input_param = [message, model_choice, chat_his, chat_bot, System_Prompt, 
                   Context_length, Temperature,max_tokens,top_p,frequency_penalty,
                   presence_penalty]
    output_param = [chat_bot, usr_msg, chat_his]

    # chatbot button event
    message.submit(deliver,
                   input_param, 
                   output_param, 
                   queue=False
                   ).success(stream,
                             [chat_bot,chat_his]
                             ,chat_bot
                             ).success(update_conversation_to_json,
                                       [chat_name,chat_bot])
    send.click(deliver,
               input_param, 
               output_param, 
               queue=False
               ).success(stream,
                         [chat_bot,chat_his],
                         chat_bot
                         ).success(update_conversation_to_json,
                                   [chat_name,chat_bot])
    clear.click(rst_mem,
                inputs=chat_his,
                outputs=chat_his
                ).success(update_conversation_to_json,
                          [chat_name,chat_bot])
    # export_his.click(export_to_markdown,[chat_bot,chat_name])

    message.submit(lambda: gr.Textbox(value=''), [],[message])
    send.click(lambda: gr.Textbox(value=''), [],[message])
    
    # chat_file button event
    file.upload(upload_file,inputs=[file,split_tmp],outputs=[split_tmp,file],show_progress="full").then(cal_token_cost,[split_tmp],[estimate_cost])
    file.clear(lambda:gr.Textbox(value=''),[],[estimate_cost])
    refresh_file_cost.click(lambda:gr.Text(),[],[estimate_cost]).then(lambda:gr.File(),[],[file]).then(lambda:gr.Text(),[],[estimate_cost])
    chat_with_file.click(ask_file,
                         inputs=[chat_bot,message,file_answer,model_choice,
                                 sum_type,vector_path,file_list,filter_choice],
                         outputs=[chat_bot,file_answer]
                         ).then(file_ask_stream,
                                [chat_bot,file_answer],
                                [chat_bot]
                                ).success(update_conversation_to_json,
                                          [chat_name,chat_bot])
    summarize.click(summarize_file,
                    inputs=[split_tmp,chat_bot,model_choice,sum_type],
                    outputs=[sum_result,chat_bot]
                    ).then(sum_stream,
                           [sum_result,chat_bot],
                           [chat_bot]
                           ).success(update_conversation_to_json,
                                     [chat_name,chat_bot])

    chat_with_file.click(lambda: gr.Textbox(value=''), [],[message])
    summarize.click(lambda: gr.Textbox(value=''), [],[message])

    # Manage vectorstore event
    create_vec_but.click(create_vectorstore,inputs=[vector_path])
    load_vec.click(load_vectorstore,inputs=[vector_path],outputs=[vector_content,file_list])
    #file_list.change(refresh_file_list,inputs=[vector_content],outputs=file_list)
    add_file.click(add_file_in_vectorstore,inputs=[vector_path,split_tmp,file],outputs=[vector_content,file_list]).then(load_vectorstore,inputs=[vector_path],outputs=[vector_content,file_list])
    delete_file.click(delete_flie_in_vectorstore,inputs=file_list).then(load_vectorstore,inputs=[vector_path],outputs=[vector_content,file_list])

demo.queue().launch(inbrowser=True,debug=True,show_api=False
                    #auth=[("admin","123456")],auth_message="欢迎使用 GPT-Gradio-Agent ,请输入用户名和密码"
                    )