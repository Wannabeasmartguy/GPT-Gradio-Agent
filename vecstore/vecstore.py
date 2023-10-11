
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import gradio as gr
import pandas as pd

def create_vectorstore(persist_vec_path:str):
    '''
    Create vectorstore.
    '''
    embeddings = OpenAIEmbeddings()

    global vectorstore
    vectorstore = Chroma(persist_directory=persist_vec_path,embedding_function=embeddings)
    vectorstore.persist()
    return vectorstore

def add_file_in_vectorstore(persist_vec_path:str, 
                            split_docs:list,
                            file_obj   # get it from 'file' (gr.file)
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
    #else:
    #    vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
    
    # Before we add file, we should detect if there is a file with the same name
    import os
    
    # New file's name
    file_absolute_path = file_obj.name
    print(file_absolute_path)
    file_name = os.path.basename(file_absolute_path)
    
    vct_store = vectorstore.get()
    unique_sources = set(vct_store['metadatas'][i]['source'] for i in range(len(vct_store['metadatas'])))
    
    # List of names of files in kownledge base
    vec_file_names = [source.split('/')[-1].split('\\')[-1] for source in unique_sources]

    # Compare file_name with vec_file_names
    if file_name in vec_file_names: 
        raise gr.Error('File already exists in vectorstore.')
    
    # If file is already exist, it won't be added repeatedly
    vectorstore.add_documents(documents=split_docs[-1])

# def save_vectorstore(vectorstore:Chroma):
#     '''
#     Save vectorstore.
#     '''
#     vectorstore.persist()

def delete_flie_in_vectorstore(file_list
                               ):
    '''
    Get the file's ids first, then delete by vector IDs.
    '''
    # Specify the target file
    metadata = vectorstore.get()
    # Initialize an empty list to store the ids
    ids_for_target_file = []

    # Loop over the metadata
    for i in range(len(metadata['metadatas'])):
        # Check if the source matches the target file
        # We only compare the last part of the path (the actual file name)
        if metadata['metadatas'][i]['source'].split('/')[-1].split('\\')[-1] == file_list:
            # If it matches, add the corresponding id to the list
            ids_for_target_file.append(metadata['ids'][i])

    # print("IDs for target file:", ids_for_target_file)
    vectorstore.delete(ids=ids_for_target_file)


def load_vectorstore(persist_vec_path:str):
    '''
    Load vectorstore, and trun the files' name to dataframe.
    '''
    global vectorstore

    if persist_vec_path:
        vectorstore = Chroma(persist_directory=persist_vec_path, 
                             embedding_function=OpenAIEmbeddings())
    else:
        vectorstore = Chroma(embedding_function=OpenAIEmbeddings())

    vct_store = vectorstore.get()
    unique_sources = set(vct_store['metadatas'][i]['source'] for i in range(len(vct_store['metadatas'])))

    # Merge duplicate sources
    merged_sources = ', '.join(unique_sources)

    # Extract actual file names
    file_names = [source.split('/')[-1].split('\\')[-1] for source in unique_sources]

    df = pd.DataFrame(file_names, columns=['文件名称'])

    return df

def refresh_file_list(df):
    file_list = df['文件名称'].tolist()
    gr.Info('Successfully update kowledge base.')
    return gr.Dropdown.update(choices=file_list)