import gradio as gr
import openai
import time
import os
from dotenv import load_dotenv
import pandas

# Customized Modules
from vecstore.vecstore import * 
from vecstore.Agent import *
from gga_utils.common import *
from gga_utils.theme import *
from vecstore.template import *
from gga_utils.vec_utils import *

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

# Initialize language
i18n = I18nAuto()  
# <---------- set environmental parameters --------->

def model_token_correct(model_choice:str):
    '''Different model has different max tokens, this is to correct the max_token slider right.'''
    model_maxtoken_dic = {
        "gpt-35-turbo":3000,
        "gpt-35-turbo-16k":15000,
        "gpt-4":7000,
        "gpt-4-32k":30000,
        "gpt-4-turbo-pr":7000
    }
    return model_maxtoken_dic[model_choice]

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
    text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=0)
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
    ref_result = get_accordion(res=file_answer[0],response=file_answer[0]["answer"])
    file_ask_history_list[-1][1] = ""
    for character in bot_message:
        file_ask_history_list[-1][1] += character
        time.sleep(0.02)
        yield file_ask_history_list
    file_ask_history_list[-1][1] += ref_result
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
    with gr.Row():
        with gr.Column(elem_id="history"):
            with gr.Row():
                add_dialog = gr.ClearButton(
                    components=[chat_his],
                    icon=r"icon\add_dialog.png",
                    #variant="primary",
                    value=i18n("New Dialog"),
                    min_width=5,
                    elem_id="btn_transparent",
                    size="sm"
                )
                delete_dialog = gr.Button(
                    icon=r"icon\delete_dialog.png",
                    value=i18n("Delete Dialog"),
                    min_width=5,
                    elem_id="btn_transparent",
                    size="sm",
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
        with gr.Column(scale=4):
            model_choice = gr.Radio(choices=["gpt-35-turbo","gpt-35-turbo-16k","gpt-4","gpt-4-32k","gpt-4-turbo-pr"],
                                    value="gpt-35-turbo",
                                    label=i18n("Model"),
                                    info=i18n("Model info"),)
            with gr.Tab(label=i18n("ChatInterface")):
                with gr.Group():
                    chat_name = gr.Textbox(label=i18n("Chatbot name"),
                                            interactive=True,
                                            value=get_last_conversation_name(),
                                            info=i18n("Chatbot info"))
                    chat_bot = gr.Chatbot(height=600,
                                        value=get_last_conversation_content(),
                                        show_label=False,
                                        show_copy_button=True,
                                        bubble_full_width=False,
                                        render_markdown=True,)
                with gr.Row():
                    message = gr.Textbox(label=i18n("Input your prompt"),
                                        info=i18n("'Shift + Enter' to begin an new line. Press 'Enter' can also send your Prompt to the LLM."),
                                        scale=7)
                    export_his = gr.Button(value=i18n("Export Chat History"),scale=1)
                with gr.Row():
                    chat_with_file = gr.Button(value=i18n("Chat with file (Valid for 📁)"))
                    send = gr.Button(i18n("Send"),variant='primary',elem_id="btn",scale=2)
                with gr.Row():
                    delete_latest_round_button = gr.Button(i18n("Delete previous round"),scale=1,size="sm")
                    regenerate_button = gr.Button(i18n("Regenerate"),scale=1,size="sm")
                    clear = gr.ClearButton([message, chat_bot,chat_his],value=i18n("Clear"),scale=1)

            with gr.Tab(label=i18n("Knowledge Base Info Interface")):
                kb_vector_content = gr.DataFrame(visible=False,interactive=False,)
                with gr.Row():
                    kb_path = gr.Textbox(label=i18n("Knowledge Base path"),
                                         scale=3)
                    kb_file_list = gr.Dropdown(interactive=True,        # The contents are exactly the same as file_list
                                                # allow_custom_value=True,
                                                label=i18n("File list"),
                                                scale=3)
                    with gr.Column(scale=1):
                        refresh_kb_info = gr.Button(value=i18n("Refresh"))
                        advance_kb_info = gr.Checkbox(label=i18n("Show document details"))
                kb_info = gr.HTML(value=i18n("Knowledge base not loaded"))
            with gr.Tab(label="Dall-E"):
                with gr.Row():
                    t2p_model = gr.Radio(choices=["Dall-E-3"],
                                         value="Dall-E-3",
                                         label=i18n("Model"),
                                         scale=1)
                    pic_gen_prompt = gr.Textbox(label=i18n("Input your prompt"),
                                                info=i18n("'Shift + Enter' to begin an new line. "),
                                                scale=5)
                    pic_gen_button = gr.Button(value=i18n("Generate"),
                                               variant="primary",
                                               elem_id="btn",
                                               scale=1)
                img = gr.Image(height=400,
                               interactive=False)
                open_dir = gr.Button(value=i18n("Open output directory"))

        with gr.Column():
            with gr.Tab(i18n("Chat")):
                with gr.Row():
                    with gr.Column():
                        with gr.Accordion(i18n("Commom Setting"),
                                          elem_id="Accordion"):
                            System_Prompt = gr.Textbox("You are a helpful AI.", label=i18n("System Prompt"),
                                                    info=i18n("'Shift + Enter' to begin an new line."))
                            Context_length = gr.Slider(0, 32, value=4, step=1, label=i18n("Context length"),
                                                    info=i18n("The number of historical messages carried per request"))                    
            
                        with gr.Accordion(i18n("Additional Setting"),
                                          elem_id="Accordion"):
                            max_tokens = gr.Slider(0, model_token_correct("gpt-35-turbo"), value=1200, step=1, label="max_tokens",
                                                info=i18n("Maximum number of tokens carrying context interactions."))
                            Temperature = gr.Slider(0, 2, value=0.5, step=0.1, label="Temperature",
                                                    info=i18n("Randomness: the larger the value, the more random the response is"))
                            top_p = gr.Slider(0, 1, value=1, step=0.1, label="top_p",
                                            info=i18n("Nuclear sampling: Similar to randomness, but not modified with randomness"))
                            frequency_penalty = gr.Slider(-2, 2, value=0, step=0.1, label=i18n("frequency_penalty"),
                                                        info=i18n("Frequency penalty: the larger the value, the less likely it is to be a repeated word"))
                            presence_penalty = gr.Slider(-2, 2, value=0, step=0.1, label=i18n("presence_penalty"),
                                                        info=i18n("Topic freshness: the larger the value, the more likely it is to expand to new topics"))
            with gr.Tab("RAG"):
                split_tmp = gr.State(['0'])
                sum_result = gr.State()
                # set a element to aviod indexerror
                file_answer = gr.State(['0']) 
                
                with gr.Column():
                    with gr.Accordion(label=i18n("RAG Basic Operation"),
                                      elem_id="Accordion"):
                        with gr.Group():
                            file = gr.File(label=i18n("The file you want to chat with"),
                                        file_types=[".eml", ".html", ".json", ".md", ".msg", ".rst", ".rtf", ".txt", ".xml",# Plaintext
                                                    # ".jpeg", ".png",# images
                                                    ".csv", ".doc", ".docx", ".epub", ".odt", ".pdf", ".ppt", ".pptx", ".tsv", ".xlsx"# Documents
                                                    ],
                                        height=150)
                            summarize = gr.Button(value=i18n("Summarize file content"))
                            with gr.Row():
                                estimate_cost = gr.Text(label=i18n("Estimated cost:"), 
                                                        info=i18n("Estimated cost of embed file"),
                                                        scale=2)
                                refresh_file_cost = gr.Button(value=i18n("Refresh file and estimate cost"),
                                                            scale=1)
                        with gr.Group():
                            vector_path = gr.Text(label=i18n("Knowledge base save path"),
                                                info=i18n("Choose the folder you want to save, and PASTE THE ABSOLUTE PATH here"))
                            with gr.Row():
                                vector_content = gr.DataFrame(#label="Knowledge Base Document Catalog",
                                                            value = pd.DataFrame(columns=['文件名称']),
                                                            visible=False,
                                                            interactive=False,
                                                            )
                                file_list = gr.Dropdown(interactive=True,
                                                        # allow_custom_value=True,
                                                        label=i18n("File list"))
                        with gr.Column():
                            create_vec_but = gr.Button(value=i18n("Create a new knowledge base 📁"))
                            load_vec = gr.Button(value=i18n("Load your 📁 "),variant='primary',elem_id="btn")
                            with gr.Row():
                                add_file = gr.Button(value=i18n("Add it (The file uploaded) to 📁"),
                                                    scale=1)
                                delete_file = gr.Button(value=i18n("Delete it (Selected in dropdown) from 📁"),
                                                        scale=1)  
                    with gr.Accordion(i18n("File chat setting"),
                                      open=False,
                                      elem_id="Accordion"):
                        filter_choice = gr.Radio(choices=["All", "Selected file"],
                                                value="All",
                                                label=i18n("Search scope"),
                                                info=i18n("“All” means whole knowledge base;“Selected file” means the file selected in dropdown"))
                        sum_type = gr.Radio(choices=[(i18n("small file"),"stuff"),
                                                     (i18n("large file(refine)"),"refine"),
                                                     (i18n("large file(map reduce)"),"map_reduce"),
                                                     (i18n("large file(map rerank, for chat)"),"map_rerank")],
                                            value="refine",
                                            label=i18n("File size type"),
                                            info=i18n("Only works on the file selected in the file box. If the number of words to be summarized is large, select 'lagre size' (selecting 'small size' may result in exceeding the GPT's maximum Token)."))
            with gr.Tab("Agent"):
                with gr.Tab(i18n("Web Request")):
                    sum_url = gr.Textbox(label=i18n("URL"),
                                         info=i18n("Paste the link to the page you want to request here."))
                    web_template = gr.Textbox(label=i18n("Prompt Template"),
                                              info=i18n("Input the template you want to use here."))
                    sum_url_button = gr.Button(value=i18n("Request URL"),
                                               variant='primary',
                                               elem_id="btn",
                                               scale=2)
                    template_example = gr.Examples([sum_wechat_gzh,
                                                    sina_test],
                                                    inputs=[web_template],
                                                    )

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
                                           [chat_bot]).success(reload_memory,
                                                               [chat_bot,Context_length]
                                                               ).success(lambda: gr.Info(i18n("Load dialog memory success!")))

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

    # update model max_token
    model_choice.change(lambda model_choice: gr.Slider(maximum=model_token_correct(model_choice),value=1200),
                                                       inputs=[model_choice], 
                                                       outputs=[max_tokens]
                                                       )

    # Knowledge base refresh button event
    refresh_kb_info.click(load_vectorstore,
                          inputs=[kb_path],
                          outputs=[kb_vector_content,kb_file_list]
                          ).success(get_chroma_info,
                                    [kb_path,kb_file_list,advance_kb_info],
                                    [kb_info])
    
    kb_file_list.change(get_chroma_info,
                        [kb_path,kb_file_list,advance_kb_info],
                        [kb_info])
    
    advance_kb_info.select(get_chroma_info,
                           [kb_path,kb_file_list,advance_kb_info],
                           [kb_info])
    
    # chatbot button event
    message.submit(reload_memory,
                   [chat_bot,Context_length],
                   ).success(deliver,
                             input_param, 
                             output_param, 
                             queue=False
                             ).then(lambda: gr.Textbox(value=''), 
                                     [],
                                     [message]
                                     ).success(stream,
                                        [chat_bot,chat_his]
                                        ,chat_bot
                                        ).success(update_conversation_to_json,
                                                [chat_name,chat_bot])
    send.click(reload_memory,
               [chat_bot,Context_length],
               ).success(deliver,
                         input_param, 
                         output_param, 
                         queue=False
                         ).then(lambda: gr.Textbox(value=''), [],[message]
                                ).success(stream,
                                         [chat_bot,chat_his],
                                         chat_bot
                                         ).success(update_conversation_to_json,
                                                  [chat_name,chat_bot])
    
    regenerate_button.click(remove_last_chat,
                            [chat_his,chat_bot],
                            [chat_his,chat_bot,message]
                            ).success(lambda: gr.Button(interactive=False),[],[send]
                                     ).success(lambda: gr.Button(interactive=False),[],[chat_with_file]
                                              ).success(reload_memory,
                                                        [chat_bot,Context_length],
                                                        ).success(deliver,
                                                                 input_param, 
                                                                 output_param, 
                                                                 queue=False
                                                                 ).then(lambda: gr.Textbox(value=''), [],[message]
                                                                        ).success(stream,
                                                                                [chat_bot,chat_his],
                                                                                chat_bot
                                                                                ).success(lambda: gr.Button(interactive=True),[],[send]
                                                                                          ).success(lambda: gr.Button(interactive=True),[],[chat_with_file]
                                                                                                    ).success(update_conversation_to_json,
                                                                                                              [chat_name,chat_bot])
    
    clear.click(rst_mem,
                inputs=chat_his,
                outputs=chat_his
                ).success(update_conversation_to_json,
                          [chat_name,chat_bot])
    
    delete_latest_round_button.click(remove_last_chat,
                                     [chat_his,chat_bot],
                                     [chat_his,chat_bot]
                                     ).success(update_conversation_to_json,
                                              [chat_name,chat_bot]).success(lambda: gr.Info(i18n("Successfully delete current round!")))
    
    export_his.click(export_to_markdown,[chat_bot,chat_name])
    
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
    load_vec.click(load_vectorstore,
                   inputs=[vector_path],
                   outputs=[vector_content,file_list]
                   ).then(lambda vector_path:gr.Textbox(value=vector_path),
                          [vector_path],
                          [kb_path]
                          ).then(load_vectorstore,
                                 [vector_path],
                                 [kb_vector_content,kb_file_list])
    
    #file_list.change(refresh_file_list,inputs=[vector_content],outputs=file_list)
    add_file.click(add_file_in_vectorstore,inputs=[vector_path,split_tmp,file],outputs=[vector_content,file_list]).then(load_vectorstore,inputs=[vector_path],outputs=[vector_content,file_list])
    delete_file.click(delete_flie_in_vectorstore,inputs=file_list).then(load_vectorstore,inputs=[vector_path],outputs=[vector_content,file_list])

    # Agent button event
    sum_url_button.click(url_request_chain,
                         inputs=[model_choice,sum_url,chat_bot,web_template],
                         outputs=[chat_bot]).success(update_conversation_to_json,
                                                      [chat_name,chat_bot])

demo.queue().launch(inbrowser=True,debug=True,show_api=False
                    #auth=[("admin","123456")],auth_message="欢迎使用 GPT-Gradio-Agent ,请输入用户名和密码"
                    )