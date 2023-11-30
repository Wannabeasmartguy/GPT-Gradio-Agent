from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA,ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from openai.error import InvalidRequestError
import openai
import gradio as gr
import pandas as pd
import tiktoken
import gradio as gr
from gga_utils.common import *

i18n = I18nAuto()  

global chat_memory
chat_memory = ConversationBufferMemory(memory_key="chat_memory", return_messages=True)

def _init():
    global vec_store
    vec_store = Chroma()

def convert_messages(messages:list):
    """
    Converts messages to the format expected by the chat model.
    """
    converted_messages = []
    for message in messages:
        if message.type == "human":
            converted_message = {"role": "user", "content": message.content}
        elif message.type == "ai":
            converted_message = {"role": "assistant", "content": message.content}
        else:
            # Handle unknown message type
            continue
        converted_messages.append(converted_message)
    if converted_messages == []:
        return None
    return converted_messages

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
            presence_penalty:float,
            #chat_memory:ConversationBufferMemory
            ):
    '''
    Response function for chat-only
    '''
    global chat_memory
    if chat_memory is None:
        chat_memory = ConversationBufferMemory(memory_key="chat_memory", return_messages=True)
    
    # Load memory first
    memory_tmp = chat_memory.load_memory_variables({})["chat_memory"]
    
    # Convert to request format
    chat_history = convert_messages(memory_tmp)
    # chat_history.clear()
    # chat_history.extend(convert_messages(memory_tmp))

    # Avoid empty input
    if message == "":
        raise gr.Error("Please input a message")

    # System Prompt and User Prompt
    if system:
        system_input = {
            "role": "system",
            "content": system
        }
        if chat_history == []:
            chat_history.append(system_input)
        else:
            if chat_history == None:
                chat_history = []
                chat_history.append(system_input)
            else:
                chat_history.insert(0,system_input)
 
    user_input = {
        "role": "user",
        "content": message
    }

    chat_history.append(user_input)

    # Trim the context length first
    if (len(chat_history)-1 > context_length) and len(chat_history)>3:
        chat_history = [chat_history[0]]+chat_history[-context_length:]

    if context_length == 0:
        # If context_length == 0,clean up chat_history
        try:
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
        except InvalidRequestError:
            raise gr.Error(i18n("Max_token has exceeded the maximum value, please shorten the text or reduce the max_token setting."))
    else:
        try:
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
        except InvalidRequestError:
            raise gr.Error(i18n("Max_token has exceeded the maximum value, please shorten the text or reduce the max_token setting."))
    reply = response.choices[0].message.content
    chat_history_list.append([message,None])
    chat_memory.save_context({"input": message},{"output": reply})

    memory_tmp = chat_memory.load_memory_variables({})["chat_memory"]
    chat_history = convert_messages(memory_tmp)
    
    return chat_history_list,message,chat_history

def rst_mem(chat_his:list):
    '''
    Reset the chatbot memory(chat_his).
    '''
    chat_his = []
    if chat_memory != None:
        chat_memory.clear()
    return chat_his

# Manipulating Vector Databases
def create_vectorstore(persist_vec_path:str):
    '''
    Create vectorstore.
    '''
    if persist_vec_path == "":
        raise gr.Error("Please provide a path to persist the vectorstore.")
    
    import os
    if os.path.isabs(persist_vec_path):
        embeddings = OpenAIEmbeddings()

        # global vectorstore
        vectorstore = Chroma(persist_directory=persist_vec_path,embedding_function=embeddings)
        vectorstore.persist()
    else:
        raise gr.Error("The path is not valid.")
    
    return vectorstore

def add_file_in_vectorstore(persist_vec_path:str, 
                            split_docs:list,
                            file_obj,   # get it from 'file' (gr.file)
                            progress=gr.Progress()
                            ):
    '''
    Add file to vectorstore.
    '''

    if file_obj == None:
        raise gr.Error("You haven't chosen a file yet.")

    if persist_vec_path:
        global vectorstore
        vectorstore = Chroma(persist_directory=persist_vec_path, 
                             embedding_function=OpenAIEmbeddings())
    else:
        raise gr.Error("You haven't chosen a knowledge base yet.")
    
    # Before we add file, we should detect if there is a file with the same name
    import os
    
    # New file's name
    file_absolute_path = file_obj.name
    print(file_absolute_path)
    file_name = os.path.basename(file_absolute_path)
    
    vct_store = vectorstore.get()
    unique_sources = set(vct_store['metadatas'][i]['source'] for i in range(len(vct_store['metadatas'])))
    progress(0.3, desc="Updating knowledge base...")
    
    # List of names of files in kownledge base
    vec_file_names = [source.split('/')[-1].split('\\')[-1] for source in unique_sources]

    # Compare file_name with vec_file_names
    if file_name in vec_file_names: 
        raise gr.Error('File already exists in vectorstore.')
    
    # If file is already exist, it won't be added repeatedly
    vectorstore.add_documents(documents=split_docs[-1])
    progress(1, desc="Adding the file to the knowledge base...")
    return gr.DataFrame(),gr.Dropdown()

# def save_vectorstore(vectorstore:Chroma):
#     '''
#     Save vectorstore.
#     '''
#     vectorstore.persist()

def delete_flie_in_vectorstore(file_list,
                               progress=gr.Progress()
                               ):
    '''
    Get the file's ids first, then delete by vector IDs.
    '''
    # Specify the target file
    try:
        metadata = vectorstore.get()
    except NameError as n:
        raise gr.Error('Vectorstore is not initialized.')

    # Initialize an empty list to store the ids
    ids_for_target_file = []

    # Loop over the metadata
    for i in range(len(metadata['metadatas'])):
        # Check if the source matches the target file
        # We only compare the last part of the path (the actual file name)
        if metadata['metadatas'][i]['source'].split('/')[-1].split('\\')[-1] == file_list:
            # If it matches, add the corresponding id to the list
            ids_for_target_file.append(metadata['ids'][i])

    progress(0.9, desc="Document comparison in progress...")

    # print("IDs for target file:", ids_for_target_file)
    try:
        vectorstore.delete(ids=ids_for_target_file)
        progress(1, desc="File deleting...")
        gr.Info("The selected file has been deleted")
    except ValueError as v:
        raise gr.Error('File does not exist in vectorstore.')
    return


def load_vectorstore(persist_vec_path:str):
    '''
    Load vectorstore, and trun the files' name to dataframe.
    '''
    global vectorstore

    if persist_vec_path:
        vectorstore = Chroma(persist_directory=persist_vec_path, 
                             embedding_function=OpenAIEmbeddings())
    else:
        raise gr.Error("You didn't provide an absolute path to the knowledge base")

    try:
        vct_store = vectorstore.get()
        unique_sources = set(vct_store['metadatas'][i]['source'] for i in range(len(vct_store['metadatas'])))

        # Merge duplicate sources
        merged_sources = ', '.join(unique_sources)

        # Extract actual file names
        file_names = [source.split('/')[-1].split('\\')[-1] for source in unique_sources]

        df = pd.DataFrame(file_names, columns=['文件名称'])

        gr.Info('Successfully load kowledge base.')
        return df,gr.Dropdown(value=file_names[0],choices=file_names)
    except IndexError:
        gr.Info('No file in vectorstore.')
        return df,gr.Dropdown(choices=[])

def refresh_file_list(df):
    '''
    abandon temporarily
    '''
    file_list = df['文件名称'].tolist()
    gr.Info('Successfully update kowledge base.')
    return gr.Dropdown.update(choices=file_list)

def ask_file(file_ask_history_list:list,
            question_prompt: str,
            file_answer:list,
            model_choice:str,
            sum_type:str,
            persist_vec_path,
            file_list,
            filter_type:str,
            ):
    '''
    send splitted file to LLM
    '''
    global chat_memory 
    if chat_memory == None:
        chat_memory = ConversationBufferMemory(memory_key="chat_memory", return_messages=True)

    llm = AzureChatOpenAI(model=model_choice,
                    openai_api_type="azure",
                    deployment_name=model_choice, 
                    temperature=0.7)
    
    source_data = vectorstore.get()
    filter_goal = find_source_paths(file_list,source_data)

    if persist_vec_path != None:
        # Codes here in "if" may be deleted or modified later
        if filter_type == "All":    
            # unselect file: retrieve whole knowledge base
            try:
                chat_history_re = chat_memory.load_memory_variables({})['chat_memory']
                qa = ConversationalRetrievalChain.from_llm(
                                                            llm=llm,
                                                            retriever=vectorstore.as_retriever(search_type="mmr"),
                                                            # chain_type=sum_type,
                                                            verbose=True,
                                                            return_source_documents=True,
                                                        )
                result = qa({"question": question_prompt,"chat_history": chat_history_re})
                chat_memory.save_context({"input": result["question"]},{"output": result["answer"]})
            except (NameError):
                raise gr.Error("You have not load kownledge base yet.")
        elif filter_type == "Selected file":
            # only selected one file
            # Retrieve the specified knowledge base with filter
            chat_history_re = chat_memory.load_memory_variables({})['chat_memory']
            qa = ConversationalRetrievalChain.from_llm(
                                                        llm=llm,
                                                        retriever=vectorstore.as_retriever(search_type="mmr",search_kwargs={"filter":{"source":filter_goal[0]}}),
                                                        # chain_type=sum_type,
                                                        verbose=True,
                                                        # memory=chat_memory,
                                                        return_source_documents=True,
                                                    )
            
            # get chain's result
            result = qa({"question": question_prompt,"chat_history": chat_history_re})
            chat_memory.save_context({"input": result["question"]},{"output": result["answer"]})

        usr_prob = result["question"]
    # if there is no file, let it become a common chat model
    else:
        gr.Info("You don't select your knowledge base, so the result is presented by base model.")
        result = llm(question_prompt)+"\n引用文档："
        usr_prob = question_prompt
    file_answer[0] = result
    file_ask_history_list.append([usr_prob,None])
    return file_ask_history_list,file_answer

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

def find_source_paths(filename:str, data:dict):
    """
    Find the source paths of the files in the knowledge base.
    return --> list
    """
    paths = []
    for metadata in data['metadatas']:
        source = metadata.get('source')
        if source and filename in source and source not in paths:
            paths.append(source)
    return paths

def calculate_and_display_token_count(input_text:str,model_name:str):
    '''
    Calculate the token that embedding needs to be consumed, for being called
    '''
    # model name or encode type
    encoder = tiktoken.encoding_for_model(model_name)   # model name
    # encoder = tiktoken.get_encoding("cl100k_base")    # encode type

    encoded_text = encoder.encode(input_text)
    token_count = len(encoded_text)
    pay_for_token = (token_count/1000) * 0.002
    
    # print(f"输入的文本: '{input_text}'")
    # print(f"对应的编码: {encoded_text}")
    # print(f"Token数量: {token_count}")
    # print("预计消耗费用: $ %0.5f\n"%pay_for_token)
    return pay_for_token

def cal_token_cost(split_docs,model_name="text-embedding-ada-002"):
    '''
    Calculate the token that embedding needs to be consumed, for operation
    '''
    cost = 0
    try:
        for i in split_docs[-1]:
            paid_per_doc = calculate_and_display_token_count(input_text=i.page_content,model_name=model_name)
            cost += paid_per_doc
        return gr.Text("预计消耗费用: $ %0.5f"%cost)
    except AttributeError:
        raise gr.Error("Cost calculating failed")
    
def get_accordion(res, # 获得的 LLM 响应
                  response,# 需要添加引用的原回答
                  font_size=2, # 参考原文字体大小
                  head_acc=50):# 能直接看到的文字数目
    '''
    Reference to the original text in accordian form
    '''
    x = res['source_documents']
    refer_result = '\n\nSource:\n'
    for i in x:
        title = i.page_content[:head_acc].replace("\n", ' ').replace("<br>", ' ').replace("<p>", ' ').replace("\r", ' ')
        content = i.page_content
        refer_result += f"""<details><summary><font size="{font_size}">{title}</font></summary><font size="{font_size}">{content}</font></details>"""
    return refer_result