
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import pandas as pd

def create_vectorstore(persist_vec_path:str):
    '''
    Create vectorstore.
    '''
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(persist_directory=persist_vec_path,embedding_function=embeddings)
    vectorstore.persist()
    return vectorstore

def add_file_in_vectorstore(persist_vec_path:str, 
                            split_docs:list):
    '''
    Add file to vectorstore.
    '''
    vectorstore = Chroma(persist_directory=persist_vec_path,embedding_function=OpenAIEmbeddings())
    vectorstore.add_documents(documents=split_docs[-1])

# def save_vectorstore(vectorstore:Chroma):
#     '''
#     Save vectorstore.
#     '''
#     vectorstore.persist()

def delete_flie_in_vectorstore(persist_vec_path:str):
    '''
    Get the file's ids first, then delete by vector IDs.
    '''
    vectorstore = Chroma(persist_directory=persist_vec_path, 
                         embedding_function=OpenAIEmbeddings())
    # Specify the target file
    metadata = vectorstore.get()
    # Initialize an empty list to store the ids
    ids_for_target_file = []

    # Loop over the metadata
    for i in range(len(metadata['metadatas'])):
        # Check if the source matches the target file
        # We only compare the last part of the path (the actual file name)
        if metadata['metadatas'][i]['source'].split('/')[-1].split('\\')[-1] == metadata:
            # If it matches, add the corresponding id to the list
            ids_for_target_file.append(metadata['ids'][i])

    # print("IDs for target file:", ids_for_target_file)
    vectorstore.delete(ids=ids_for_target_file)


def load_vectorstore(persist_vec_path:str):
    '''
    Load vectorstore, and trun the files' name to dataframe.
    '''
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