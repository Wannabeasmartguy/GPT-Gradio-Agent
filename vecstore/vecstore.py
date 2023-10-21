from langchain.chains.summarize import load_summarize_chain
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
import gradio as gr
import pandas as pd
import tiktoken

def _init():
    global vec_store
    vec_store = Chroma()

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

    vct_store = vectorstore.get()
    unique_sources = set(vct_store['metadatas'][i]['source'] for i in range(len(vct_store['metadatas'])))

    # Merge duplicate sources
    merged_sources = ', '.join(unique_sources)

    # Extract actual file names
    file_names = [source.split('/')[-1].split('\\')[-1] for source in unique_sources]

    df = pd.DataFrame(file_names, columns=['文件名称'])

    gr.Info('Successfully load kowledge base.')
    return df,gr.Dropdown(choices=file_names)

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
                qa = RetrievalQA.from_chain_type(llm=llm, chain_type=sum_type, 
                                                    retriever=vectorstore.as_retriever(search_type="mmr"), 
                                                    return_source_documents=True)
                result = qa({"query": question_prompt})
            except (NameError):
                raise gr.Error("You have not load kownledge base yet.")
        elif filter_type == "Selected file":
            # only selected one file
            # Retrieve the specified knowledge base with filter
            qa = RetrievalQA.from_chain_type(llm=llm, chain_type=sum_type, 
                                                retriever=vectorstore.as_retriever(search_type="mmr",search_kwargs={'filter': {"source":filter_goal[0]}}), 
                                                return_source_documents=True)
            
            # get chain's result
            result = qa({"query": question_prompt})

        usr_prob = result["query"]
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