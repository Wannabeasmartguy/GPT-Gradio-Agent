from langchain.chains.summarize import load_summarize_chain
from langchain_community.vectorstores import chroma
from langchain_community.chat_models.ollama import ChatOllama
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_text_splitters.markdown import MarkdownTextSplitter
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.retrievers import ContextualCompressionRetriever,ParentDocumentRetriever,EnsembleRetriever

from huggingface_hub import snapshot_download
from typing import Literal,Union
from pathlib import Path
from openai import BadRequestError
import chromadb
from openai import AzureOpenAI
import gradio as gr
import pandas as pd
import tiktoken
import gradio as gr
import json
from gga_utils.common import *
from gga_utils.vec_utils import *
from local_llm.ollama import *
from vecstore.reranker import *
# from reranker import *
import functools
import os
from dotenv import load_dotenv
load_dotenv()


model_type_choice = ["OpenAI","Hugging Face(local)"]
openai_embedding_model = ["text-embedding-ada-002"]
local_embedding_model = ['bge-base-zh-v1.5','bge-base-en-v1.5',
                         'bge-large-zh-v1.5','bge-large-en-v1.5']
embedding_decoder_dic = {
    'model_type_choice':model_type_choice,
    'openai_embedding_model':openai_embedding_model,
    'local_embedding_model':local_embedding_model
}

i18n = I18nAuto()  

global chat_memory
chat_memory = ConversationBufferMemory(memory_key="chat_memory", return_messages=True)

client = AzureOpenAI(
  azure_endpoint = os.getenv('AZURE_OAI_ENDPOINT'), 
  api_key = os.getenv('AZURE_OAI_KEY'),  
  api_version = os.getenv('API_VERSION')
)
    
def combine_lists_to_dicts(docs, ids, metas):
    """
    将三个列表的对应元素组合成一个个字典，然后将这些字典保存在一个列表中。

    参数:
    docs (list of str): 文档名列表
    ids (list of str): id列表
    metas (list of str): 元数据列表

    返回:
    list of dict: 每个字典包含三个键值对，键分别是"documents", "ids", "metadatas"，值来自对应的列表

    示例:
    combine_lists_to_dicts(["你好","hello"], ["sabea-12","asdao-141"], ["CoT.txt","abs.txt"])
    返回 [{"documents":"你好","ids":"sabea-12","metadatas":"CoT.txt"},{"documents":"hello","ids":"asdao-141","metadatas":"abs.txt"}]
    """

    # 使用zip函数将三个列表的对应元素打包成一个个元组
    tuples = zip(docs, ids, metas)

    # 将每个元组转换为字典，然后将这些字典保存在一个列表中
    dict_lists = [{"documents": doc, "ids": id, "metadatas": meta} for doc, id, meta in tuples]

    return dict_lists

def vectorstore_init_create(persist_vec_path, 
                            embedding_model_type,
                            local_embedding_model, 
                            progress=gr.Progress()):
    '''
    Generic Vectorstore Creation Functions

    Args:
        persist_vec_path: chroma persist path.
        embedding_model_type: `OpenAI` or `Hugging Face(local)`
        local_embedding_model: If `embedding_model_type` == `OpenAI`, there's only one model, here will give a specific embedding model. 
        progress: create gradio progress bar.
    '''
    local_embedding_model_path = os.path.join("embedding model/",local_embedding_model)
    if embedding_model_type == 'Hugging Face(local)':
        try:
            embeddings = SentenceTransformerEmbeddings(model_name=local_embedding_model_path)
            vectorstore = chroma.Chroma(
                persist_directory=persist_vec_path, embedding_function=embeddings
            )
        except:
            progress(0.3, "Downloading embedding model...")
            if local_embedding_model[:3] == 'bge':
                snapshot_download(repo_id="BAAI/"+local_embedding_model,
                                    local_dir=local_embedding_model_path)
                embeddings = SentenceTransformerEmbeddings(model_name=local_embedding_model_path)
                vectorstore = chroma.Chroma(persist_directory=persist_vec_path,
                                            embedding_function=embeddings)
                
    elif embedding_model_type == 'OpenAI':
        vectorstore = chroma.Chroma(persist_directory=persist_vec_path, 
                             embedding_function=AzureOpenAIEmbeddings(
                             openai_api_type=os.getenv('API_TYPE'),
                             azure_endpoint=os.getenv('AZURE_OAI_ENDPOINT'),
                             openai_api_key=os.getenv('AZURE_OAI_KEY'),
                             openai_api_version=os.getenv('API_VERSION'),
                             azure_deployment="text-embedding-ada-002",
                            ))
    return vectorstore

def get_chroma_info(persist_path:str,
                    file_name:str,
                    advance_info:bool,
                    limit:int=10000000000):
    try:
        client = chromadb.PersistentClient(path=persist_path)
        collection_lang = client.get_collection("langchain")
    except ValueError:
        raise gr.Error(i18n("“Knowledge Base path” is empty, Please enter the path"))
    metadata_pre10 = collection_lang.peek(limit=limit)  
    
    #get data for the first <limit> files 
    documents = metadata_pre10['documents']
    ids = metadata_pre10['ids']
    metadatas = metadata_pre10['metadatas']

    chroma_data_dic = combine_lists_to_dicts(documents, ids, metadatas)
    
    kb_info_html = dict_to_html(chroma_data_dic,file_name,advance_info)
    return kb_info_html
    
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

def reload_memory(chat_bot:list[list],
                  context_length:int,
                  ):
    '''
    Applies to two situations:\n
    1. initialize the memory when client is created: to load contents in `chat_bot` as primary memory(according to `context_length`);
    2. switch the dialog: load new chat memory. 
    '''

    global chat_memory
    # Before initiate memory, clear it to make function have wider applicability
    chat_memory.clear()

    # if chat_bot is not null, and its length < context_length in setting, send it all to memery
    if (chat_bot != None or chat_bot!=[]) and len(chat_bot)<=context_length:
        for round in chat_bot:
            message = round[0]
            reply = round[1]
            chat_memory.save_context({"input": message},{"output": reply})

    # if chat_bot is not null, and its length > context_length in setting, send its latest rounds to memery
    elif (chat_bot != None or chat_bot!=[]) and len(chat_bot)>context_length:
        for round in chat_bot[-context_length:]:
            message = round[0]
            reply = round[1]
            chat_memory.save_context({"input": message},{"output": reply})

    else:
        # if chat_bot is null, that means this chatbot is a new chat or is cleared
        # nothing needs to be done
        pass

def deliver(message:str,
            chat_model_type:str,
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
    # The first time run it, need to load the contents of chat_bot into memory by context_length

    global chat_memory

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

    if chat_model_type == "OpenAI":
        if context_length == 0:
            # If context_length == 0,clean up chat_history
            try:
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=[system_input,user_input],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
            )
            except BadRequestError:
                raise gr.Error(i18n("Max_token has exceeded the maximum value, please shorten the text or reduce the max_token setting."))
        else:
            try:
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=chat_history,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    frequency_penalty=frequency_penalty,
                    presence_penalty=presence_penalty,
                )
            except BadRequestError:
                raise gr.Error(i18n("Max_token has exceeded the maximum value, please shorten the text or reduce the max_token setting."))
        reply = response.choices[0].message.content

    elif chat_model_type == "Ollama":
        if context_length == 0:
            try:
                response = send_ollama_chat_request(messages=[system_input,user_input],
                                             model=model_choice)
            except BadRequestError:
                raise gr.Error(i18n("Max_token has exceeded the maximum value, please shorten the text or reduce the max_token setting."))
        else:
            try:
                response = send_ollama_chat_request(messages=chat_history,
                                             model=model_choice)
            except BadRequestError:
                raise gr.Error(i18n("Max_token has exceeded the maximum value, please shorten the text or reduce the max_token setting."))
        reply = response["message"]["content"]

    chat_history_list.append([message,None])
    chat_memory.save_context({"input": message},{"output": reply})

    memory_tmp = chat_memory.load_memory_variables({})["chat_memory"]
    chat_history = convert_messages(memory_tmp)
    
    return chat_history_list,message,chat_history

def remove_last_chat(chat_history:list,
                     chat_bot:list[list]):
    '''Delete the latest round of conversation, fill `message` with last question.'''
    if len(chat_bot)==1:
        message = chat_bot[0][0]
    elif len(chat_bot)==0 or chat_bot==None:
        raise gr.Error(i18n("You haven't sent a message yet."))
    else:
        message = chat_bot[-1][0]

    if chat_bot is not None and len(chat_bot) > 0:
        # 删除最后一个元素
        chat_bot.pop()
    else:
        raise gr.Error(i18n("You haven't sent a message yet."))
    
    # 返回原函数的结果
    return chat_history,chat_bot,message

def rst_mem(chat_his:list):
    '''
    Reset the chatbot memory(chat_his).
    '''
    chat_his = []
    if chat_memory != None:
        chat_memory.clear()
    return chat_his

# Manipulating Vector Databases
def create_vectorstore(persist_vec_path:str,
                       embedding_model_type:Literal['OpenAI','Hugging Face(local)'],
                       embedding_model:str):
    '''
    Create vectorstore.
    '''
    if persist_vec_path == "":
        raise gr.Error("Please provide a path to persist the vectorstore.")
    
    local_embedding_model = 'embedding model/'+embedding_model

    import os
    if os.path.isabs(persist_vec_path):
        if embedding_model_type == 'OpenAI':
            embeddings = AzureOpenAIEmbeddings(
                                                openai_api_type=os.getenv('API_TYPE'),
                                                azure_endpoint=os.getenv('AZURE_OAI_ENDPOINT'),
                                                openai_api_key=os.getenv('AZURE_OAI_KEY'),
                                                openai_api_version=os.getenv('API_VERSION'),
                                                azure_deployment="text-embedding-ada-002",
                                                )
        elif embedding_model_type == 'Hugging Face(local)':
            try:
                embeddings = SentenceTransformerEmbeddings(model_name=local_embedding_model)
            except:
                # 如果 embedding model 的前三个字母是 bge ,则在 repo_id 前加上 BAAI/
                if embedding_model[:3] == 'bge':
                    snapshot_download(repo_id="BAAI/"+embedding_model,
                                      local_dir=local_embedding_model)
                    embeddings = SentenceTransformerEmbeddings(model_name=local_embedding_model)

        # global vectorstore
        vectorstore = chroma.Chroma(persist_directory=persist_vec_path,embedding_function=embeddings)
        vectorstore.persist()
    else:
        raise gr.Error("The path is not valid.")
    
    return vectorstore

def create_vec_in_specific_path(persist_vec_name:str,
                                embedding_model_type:str,
                                embedding_model:str,
                                progress=gr.Progress()):
    '''在默认路径下创建指定名称的知识库。'''
    progress(0.2, desc="正在创建本地文件夹...")
    vec_path = os.path.join(os.getcwd(), "knowledge base", persist_vec_name)
    if os.path.exists(vec_path) is False:
        os.makedirs(vec_path)
    progress(0.4, desc="正在创建知识库...")
    create_vectorstore(vec_path,embedding_model_type,embedding_model)
    progress(1, desc="知识库创建完成")
    gr.Info(i18n("Create successfully!"))

def delete_vec_in_specific_path(persist_vec_name:str):
    '''删除默认路径下指定名称的知识库'''
    import shutil
    # 删除文件夹及其中的全部文件
    tobe_delete_path = os.path.join(os.getcwd(), "knowledge base", persist_vec_name)
    #TODO: 直接删除会有知识库被占用，无法直接删除的问题，搁置
    shutil.rmtree(tobe_delete_path)
    gr.Info(I18nAuto("Delete knowledge base successfully."))

'''
操作embedding_config.json
'''
def create_kb_info_in_config(persist_vec_name:str,
                             embedding_model_type:str,
                             embedding_model:str):
    import time
    '''在embedding_config.json中创建指定知识库信息'''
    with open(os.path.join(os.getcwd(), "embedding_config.json"), "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    kb_config =  {
                    "embedding_model_type": embedding_model_type,
                    "embedding_model": embedding_model
                }
    config_dict[persist_vec_name] = kb_config
    with open(os.path.join(os.getcwd(), "embedding_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)
    time.sleep(0.5)

def delete_kb_info_in_config(persist_vec_name:str):
    '''删除embedding_config.json中的指定知识库信息'''
    with open(os.path.join(os.getcwd(), "embedding_config.json"), "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict.pop(persist_vec_name)
    with open(os.path.join(os.getcwd(), "embedding_config.json"), "w", encoding="utf-8") as f:
        json.dump(config_dict, f, ensure_ascii=False, indent=4)

def add_file_in_vectorstore(persist_vec_path:str, 
                            split_docs:list,
                            embedding_model_type:str,
                            local_embedding_model:str,
                            file_obj,   # get it from 'file' (gr.file)
                            progress=gr.Progress(),
                            ):
    '''
    Add file to vectorstore.
    '''
    # 适配早期代码的问题
    split_docs_real = split_docs[-1]

    if file_obj == None:
        raise gr.Error("You haven't chosen a file yet.")

    vectorstore =  vectorstore_init_create(persist_vec_path=persist_vec_path,
                                           embedding_model_type=embedding_model_type,
                                           local_embedding_model=local_embedding_model,
                                           progress=gr.Progress())

    # else:
    #     raise gr.Error("You haven't chosen a knowledge base yet.")
    
    # Before we add file, we should detect if there is a file with the same name
    import os
    
    vct_store = vectorstore.get()
    unique_sources = set(vct_store['metadatas'][i]['source'] for i in range(len(vct_store['metadatas'])))
    progress(0.6, desc="Updating knowledge base...")
    
    # List of names of files in kownledge base
    vec_file_names = [source.split('/')[-1].split('\\')[-1] for source in unique_sources]

    if isinstance(file_obj, list):
        # If file_obj is a list, it means that the user has uploaded multiple files.
        
        # # Duplicate detection
        # for file in file_obj:
        #     file_name = file.name.split('/')[-1].split('\\')[-1]
        #     if file_name in vec_file_names:
        #         gr.Info(f'File {file_name} already exists in vectorstore.')
        #         # 如果重复，则将这部分文件从列表中删除
        #         file_obj.remove(file)

        # Add all files to vectorstore
        file_obj_name = [file.name for file in file_obj]
        split_docs_filtered = [doc for doc in split_docs_real if doc.metadata['source'] in file_obj_name]
        vectorstore.add_documents(documents=split_docs_filtered)
        progress(1, desc="Adding the files to the knowledge base...")
    else:
        # New file's name
        file_absolute_path = file_obj.name
        print(file_absolute_path)
        file_name = os.path.basename(file_absolute_path)
        # Compare file_name with vec_file_names
        if file_name in vec_file_names: 
            raise gr.Error('File already exists in vectorstore.')
    
        # If file is already exist, it won't be added repeatedly
        vectorstore.add_documents(documents=split_docs_real)
        progress(1, desc="Adding the file to the knowledge base...")
    return gr.DataFrame(),gr.Dropdown(),gr.File(value=None)

def ParentDocumentRetrieve_workflow(raw_file:list,
                                    persist_vec_path:str,
                                    embedding_model_type:Literal['OpenAI','Hugging Face(local)'],
                                    local_embedding_model:str,
                                    chat_model_type:str,
                                    model_choice:str,
                                    query:str,
                                    progress=gr.Progress()
                                    ):
    '''
    Use ParentDocumentRetriever to complete file split and retrieval (not reranked)
    
    Args:
        raw_file: uploaded files, a list.
        persist_vec_path: chroma persist path.
        embedding_model_type: `OpenAI` or `Hugging Face(local)`
        local_embedding_model: If `embedding_model_type` == `OpenAI`, there's only one model, here will give a specific embedding model. 
        chat_model_type: `OpenAI` or  `Ollama`.
        model_choice: Specific model you have chosen.
        progress: create gradio progress bar.
        query: The prompt you submited.

    '''
    '''docs'''
    loaders = map(lambda doc:UnstructuredFileLoader(doc),raw_file)
    docs = []
    for loader in loaders:
        docs.extend(loader.load())

    '''text splitter'''
    # If use `ParentDocumentRetrieve`, have to use `RecursiveCharacterTextSplitter`
    # Using `CharacterTextSplitter` instead makes the result inferior to the regular processing.
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

    # TODO：Need to find a way to save the InMemoryStore(), where will save the connections between source docs and splitted docs.
    store = InMemoryStore()

    '''vectorstore and embedding'''
    vectorstore = vectorstore_init_create(persist_vec_path=persist_vec_path,
                                          embedding_model_type=embedding_model_type,
                                          local_embedding_model=local_embedding_model,
                                          progress=gr.Progress())
    
    '''retriever'''
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    retriever.add_documents(docs)

    '''reranker'''
    compressor = BgeRerank()
    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, base_retriever=retriever
    )

    '''llm'''
    if chat_model_type == 'OpenAI':
        llm = AzureChatOpenAI(model=model_choice,
                        openai_api_type=os.getenv('API_TYPE'),
                        azure_endpoint=os.getenv('AZURE_OAI_ENDPOINT'),
                        openai_api_key=os.getenv('AZURE_OAI_KEY'),
                        openai_api_version=os.getenv('API_VERSION'),
                        # eployment_name=os.getenv('AZURE_OAI_ENDPOINT')+ "deployments/" +'gpt-35-turbo', 
                        deployment_name=model_choice,
                        temperature=0.7)
    elif chat_model_type == 'Ollama':
        llm = ChatOllama(model=model_choice)

    kc = ConversationalRetrievalChain.from_llm(llm=llm, 
                                           retriever=compression_retriever,
                                           verbose=True,
                                           return_source_documents=True)
    result = kc.invoke({"question": query,"chat_history":[]})


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

def text_to_gr_dropdown(choice:dict # 格式如 embedding_decoder_dic
                        ):
    '''
    将 str 转换为 gr.Dropdown 。
    '''
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            embedding_model_type,embedding_model = func(*args, **kwargs)
            if embedding_model_type == 'OpenAI':
                return gr.Dropdown(value=embedding_model_type,choices=choice['model_type_choice']),gr.Dropdown(value=embedding_model,choices=choice['openai_embedding_model'])
            elif embedding_model_type == 'Hugging Face(local)':
                return gr.Dropdown(value=embedding_model_type,choices=choice['model_type_choice']),gr.Dropdown(value=embedding_model,choices=choice['local_embedding_model'])
        return wrapper
    return decorator

class KnowledgeBase:
    '''
    用于管理本地知识库的基类,一般不应直接实例化该类
    '''
    def __init__(self):
        '''
        初始化时，先读取本地的`embedding_config.json`,该json的结构如下
        {
            "Knowledge_base_a": {
                "embedding_model_type": "OpenAI",
                "embedding_model": "text-embedding-ada-002"
            },
            "Knowledge_base_b":{
                "embedding_model_type": "Hugging Face(local)",
                "embedding_model": "bge-base-zh-v1.5"
            }
        }
        '''
        if os.path.exists("embedding_config.json"):
            with open("embedding_config.json", "r", encoding='utf-8') as f:
                self.embedding_config = json.load(f)
        else:
            # 如果不存在"embedding_config.json"，则创建它
            self.embedding_config = {
                "default_empty_vec": {
                    "embedding_model_type": "OpenAI",
                    "embedding_model": "text-embedding-ada-002"
                }
            }
            with open("embedding_config.json", 'w', encoding='utf-8') as file:
                json.dump(self.embedding_config, file, ensure_ascii=False, indent=4)
            tmp_vec_path = os.path.join(os.getcwd(), "knowledge base", "default_empty_vec")
            if os.path.exists(tmp_vec_path) is False:
                os.makedirs(tmp_vec_path)
            create_vectorstore(tmp_vec_path, "OpenAI", "text-embedding-ada-002")
                
        self.knowledge_bases = list(self.embedding_config.keys())

    def reinitialize(self):
        '''重新初始化，以重载json中内容'''
        self.__init__()

    def get_embedding_model(self, knowledge_base_name:str):
        """
        根据知识库名称获取嵌入模型的分类和名称

        Args:
            knowledge_base_name: `embedding_config.json` 中保存的名称;
        
        Return: 
            `embedding_model_type`: str,`embedding_model`: str
        """
        
        if knowledge_base_name in self.knowledge_bases:
            return self.embedding_config[knowledge_base_name]["embedding_model_type"],self.embedding_config[knowledge_base_name]["embedding_model"]
        else:
            raise ValueError(f"未找到名为{knowledge_base_name}的知识库")
        
    def get_persist_vec_path(self, knowledge_base_name:str):
        '''在默认路径下按名字查找知识库，并返回知识库的路径'''
        vec_root_path = os.path.join(os.getcwd(), "knowledge base")
        vec_path = os.path.join(vec_root_path, knowledge_base_name)
        if os.path.exists(vec_path):
            return vec_path
        else:
            raise ValueError(f"未找到名为{knowledge_base_name}的知识库")

class GRKnowledgeBase(KnowledgeBase):
    '''
    将类 KnowledgeBase 输出规范化为 Gradio 的组件格式，继承自 KnowledgeBase
    '''
    @text_to_gr_dropdown(embedding_decoder_dic)
    def get_embedding_model(self, knowledge_base_name:str):
        """
        根据知识库名称获取嵌入模型的分类和名称

        Args:
            knowledge_base_name: `embedding_config.json` 中保存的名称;
        
        Return: 
            `embedding_model_type`: str, 修饰为gr.dropdown
            `embedding_model`: str, 修饰为gr.dropdown
            `choice`: list, 用于更新 Dropdown 
        """
        
        if knowledge_base_name in self.knowledge_bases:
            return self.embedding_config[knowledge_base_name]["embedding_model_type"],self.embedding_config[knowledge_base_name]["embedding_model"]
        else:
            raise ValueError(f"未找到名为{knowledge_base_name}的知识库")

def load_vectorstore(persist_vec_path:str,
                     embedding_model_type:Literal['OpenAI','Hugging Face(local)'],
                     embedding_model:str,
                     progress=gr.Progress()):
    '''
    Load vectorstore, and trun the files' name to dataframe.
    '''
    global vectorstore
    embedding_model_path = 'embedding model/'+embedding_model

    if persist_vec_path:
        if embedding_model_type == 'OpenAI':
            vectorstore = chroma.Chroma(persist_directory=persist_vec_path, 
                                embedding_function=AzureOpenAIEmbeddings(
                                                openai_api_type=os.getenv('API_TYPE'),
                                                azure_endpoint=os.getenv('AZURE_OAI_ENDPOINT'),
                                                openai_api_key=os.getenv('AZURE_OAI_KEY'),
                                                openai_api_version=os.getenv('API_VERSION'),
                                                azure_deployment="text-embedding-ada-002",
                                        ))
        elif embedding_model_type == 'Hugging Face(local)':
            try:
                embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_path)
                vectorstore = chroma.Chroma(persist_directory=persist_vec_path,
                                            embedding_function=embeddings)
            except:
                if embedding_model[:3] == 'bge':
                    progress(0.1, "Downloading BGE model...")
                    snapshot_download(repo_id="BAAI/"+embedding_model,
                                      local_dir=embedding_model_path)
                    embeddings = SentenceTransformerEmbeddings(model_name=embedding_model_path)
                    vectorstore = chroma.Chroma(persist_directory=persist_vec_path,
                                                embedding_function=embeddings)
                    progress(0.3, "Model download completed.")

    else:
        raise gr.Error(i18n("You didn't provide an absolute path to the knowledge base"))

    try:
        progress(0.5, "Loading knowledge base...")
        vct_store = vectorstore.get()
        unique_sources = set(vct_store['metadatas'][i]['source'] for i in range(len(vct_store['metadatas'])))

        # Merge duplicate sources
        merged_sources = ', '.join(unique_sources)

        # Extract actual file names
        file_names = [source.split('/')[-1].split('\\')[-1] for source in unique_sources]

        df = pd.DataFrame(file_names, columns=['文件名称'])
        progress(1, "Knowledge base loaded.")
        gr.Info(i18n("Successfully load kowledge base."))
        return df,gr.Dropdown(value=file_names[0],choices=file_names)
    except IndexError:
        gr.Info(i18n('No file in vectorstore.'))
        return df,gr.Dropdown(choices=[],allow_custom_value=False)

def refresh_file_list(df):
    '''
    abandon temporarily
    '''
    file_list = df['文件名称'].tolist()
    gr.Info('Successfully update kowledge base.')
    return gr.Dropdown.update(choices=file_list)


def filter_documents_by_source(metadatas:list[dict],
                               specified_source:list) -> List[dict]:
    """
    筛选出 metadatas 中具有指定 source 值的 Document 元素。用于没有内置 filter 参数的 Retriever doc筛选(如BM25)。

    Args:
        metadatas (list[dict]): 包含 Document 元数据的列表。
        specified_source (list): 指定的 source 值。
        
    Returns:
        list[dict]: 筛选后的 Document 元数据列表。
    """
    # 使用列表推导式筛选出 metadata 中 source 与指定值相匹配的 Document 元素
    filtered_documents = [
        doc for doc in metadatas if doc['source'] in specified_source[0]
    ]
    return filtered_documents


def create_conversational_retrieval_system(if_hybrid_retrieve:bool, 
                                           hybrid_retriever_weight:float, 
                                           if_rerank:bool, 
                                           compressor, 
                                           llm,
                                           filter_goal:list,
                                           filter_type=Literal['All', 'Selected file']):
    """
    创建并返回一个会话检索系统实例。

    Args:
        if_hybrid_retrieve (bool): 是否使用混合检索器。
        hybrid_retriever_weight (float): 混合检索器中BM25检索器的权重。
        if_rerank (bool): 是否进行重排。
        compressor : 用于实现重排序的压缩器。
        llm (object): 语言模型实例。
        filter_goal (list): 筛选目标。经过 find_source_paths() 找到的指定文档。
        filter_type (Literal['All', 'Selected file']): 筛选类型。

    Return:
    - qa (object): 会话检索系统实例。
    """

    sparse_retrieve_kwargs = {
                'k': 6
            }
    if filter_type == "All":
        vec_retrieve_search_kwargs = {
            'k': 6
        }
        metadatas_in_sparse_retrieve = vectorstore.get()["metadatas"]
        
    elif filter_type == "Selected file":
        vec_retrieve_search_kwargs = {
            'k': 6,
            "filter":{
                "source":filter_goal[0]
            }
        }
        # metadatas_in_sparse_retrieve = filter_documents_by_source(vectorstore.get()["metadatas"],filter_goal)
        metadatas_in_sparse_retrieve = vectorstore.get()["metadatas"]

    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs=vec_retrieve_search_kwargs)

    if if_hybrid_retrieve:
        try:
            bm25_retriever = BM25Retriever.from_texts(
                vectorstore.get()["documents"],
                metadatas=metadatas_in_sparse_retrieve,
                kwargs=sparse_retrieve_kwargs
            )
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, retriever],
                weights=[hybrid_retriever_weight, 1 - hybrid_retriever_weight]
            )
        except Exception as e:
            raise gr.Error("Could not import rank_bm25, please install with `pip install rank_bm25`") from e

    if if_rerank:
        if if_hybrid_retrieve:
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=ensemble_retriever
            )
        else:
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=retriever
            )
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=compression_retriever,
            verbose=True,
            return_source_documents=True,
        )
    else:
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            verbose=True,
            return_source_documents=True,
        )

    return qa


def ask_file(file_ask_history_list:list,
            question_prompt: str,
            file_answer:list,
            chat_model_type:str,
            model_choice:str,
            sum_type:str,
            persist_vec_path,
            file_list,
            filter_type:str,
            if_rerank:bool,
            if_hybrid_retrieve:bool,
            hybrid_retriever_weight:float,
            ):
    '''
    Ask question to the knowledge base.
    
    Args:
        file_ask_history_list: list, the history of the question and answer.
        question_prompt: str, the question to be asked.

    '''
    global chat_memory 
    if chat_memory == None:
        chat_memory = ConversationBufferMemory(memory_key="chat_memory", return_messages=True)

    if chat_model_type == 'OpenAI':
        llm = AzureChatOpenAI(model=model_choice,
                        openai_api_type=os.getenv('API_TYPE'),
                        azure_endpoint=os.getenv('AZURE_OAI_ENDPOINT'),
                        openai_api_key=os.getenv('AZURE_OAI_KEY'),
                        openai_api_version=os.getenv('API_VERSION'),
                        # eployment_name=os.getenv('AZURE_OAI_ENDPOINT')+ "deployments/" +'gpt-35-turbo', 
                        deployment_name=model_choice,
                        temperature=0.7)
    elif chat_model_type == 'Ollama':
        llm = ChatOllama(model=model_choice)
    
    # 如果使用 rerank ，实例化 BgeRerank
    if if_rerank:
        compressor = BgeRerank()

    source_data = vectorstore.get()
    filter_goal = find_source_paths(file_list,source_data)

    if persist_vec_path != None:
        # Codes here in "if" may be deleted or modified later
        chat_history_re = chat_memory.load_memory_variables({})['chat_memory']
        qa = create_conversational_retrieval_system(if_hybrid_retrieve=if_hybrid_retrieve,
                                                    hybrid_retriever_weight=hybrid_retriever_weight,
                                                    if_rerank=if_rerank,
                                                    compressor=compressor,
                                                    llm=llm,
                                                    filter_goal=filter_goal,
                                                    filter_type=filter_type
        )
        result = qa.invoke({"question": question_prompt,"chat_history": chat_history_re})
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
                    openai_api_type=os.getenv('API_TYPE'),
                    azure_endpoint=os.getenv('AZURE_OAI_ENDPOINT'),
                    openai_api_key=os.getenv('AZURE_OAI_KEY'),
                    openai_api_version=os.getenv('API_VERSION'),
                    # eployment_name=os.getenv('AZURE_OAI_ENDPOINT')+ "deployments/" +'gpt-35-turbo', 
                    deployment_name=model_choice,
                    temperature=0.7)
    # 创建总结链
    chain = load_summarize_chain(llm, chain_type=sum_type, verbose=True)
    
    # 执行总结链
    # 这里的 run 方法在 langchain v0.1 可用，v0.2将废除
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


def choose_text_splitter(file_path,
                         chunk_size:int=1000,
                         chunk_overlap:int=0):
    '''
    根据文件类型选择不同的文本分割器
    
    Args:
        file_path: 文件对象，单个临时文件或一个文件地址列表；
        chunk_size: 每个分块的大小，默认为1000；
        chunk_overlap: 每个分块的重叠部分，默认为0。
    '''

    #如果 file_path 是一个list
    if isinstance(file_path,list):
        file_path_list = [i.name for i in file_path]
        splitted_docs = []

        for file_path in file_path_list:
            # 根据扩展名判断文件类型
            file_ext = os.path.splitext(file_path)[1]
            if file_ext == '.pdf':
                loader = UnstructuredFileLoader(file_path)
                document = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
                split_docs = text_splitter.split_documents(document)
                splitted_docs.extend(split_docs)
            elif file_ext =='.md':
                loader = UnstructuredMarkdownLoader(file_path)
                document = loader.load()
                text_splitter = MarkdownTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
                split_docs = text_splitter.split_documents(document)
                splitted_docs.extend(split_docs)
            else:
                loader = UnstructuredFileLoader(file_path)
                document = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
                split_docs = text_splitter.split_documents(document)
                splitted_docs.extend(split_docs)
        
        return splitted_docs

    # 如果 file_path 是一个obj
    else:
        file_ext = os.path.splitext(file_path.name)[1]
        if file_ext == '.pdf':
                loader = UnstructuredFileLoader(file_path.name)
                document = loader.load()
                text_splitter = CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
                split_docs = text_splitter.split_documents(document)
        elif file_ext =='.md':
            loader = UnstructuredMarkdownLoader(file_path.name)
            document = loader.load()
            text_splitter = MarkdownTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
            split_docs = text_splitter.split_documents(document)
        else:
            loader = UnstructuredFileLoader(file_path.name)
            document = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
            split_docs = text_splitter.split_documents(document)

        return split_docs
