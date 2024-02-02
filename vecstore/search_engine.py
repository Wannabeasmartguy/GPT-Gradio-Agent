import os
import requests
import time
import gradio as gr
from gga_utils.common import *
from dotenv import load_dotenv
from openai import AzureOpenAI
from loguru import logger

load_dotenv()

i18n = I18nAuto()  

_rag_query_text = """
You are a large language AI assistant built by GGA AI. You are given a user question, and please write clean, concise and accurate answer to the question. You will be given a set of related contexts to the question, each starting with a reference number like [[citation:x]], where x is a number. Please use the context and cite the context at the end of each sentence if applicable.

Your answer must be correct, accurate and written by an expert using an unbiased and professional tone. Please limit to 1024 tokens. Do not give any information that is not related to the question, and do not repeat. Say "information is missing on" followed by the related topic, if the given context do not provide sufficient information.

Please cite the contexts with the reference numbers, in the format [citation:x]. If a sentence comes from multiple contexts, please list all applicable citations, like [citation:3][citation:5]. Other than code and specific names and citations, your answer must be written in the same language as the question.

Here are the set of contexts:

{context}

Remember, don't blindly repeat the contexts verbatim. And here is the user question:
"""

stop_words = [
    "<|im_end|>",
    "[End]",
    "[end]",
    "\nReferences:\n",
    "\nSources:\n",
    "End.",
]

BING_MKT = "en-US"
BING_SEARCH_V7_ENDPOINT = os.getenv("BING_SEARCH_URL")
subscription_key = os.getenv("BING_SUBSCRIPTION_KEY")
DEFAULT_SEARCH_ENGINE_TIMEOUT = 5
REFERENCE_COUNT = 8

class BingSearchValueError(Exception):
    pass

def search_with_bing(query: str, subscription_key: str = os.getenv("BING_SUBSCRIPTION_KEY")):
    """
    Search with bing and return the contexts.
    """
    params = {"q": query, "mkt": BING_MKT}
    response = requests.get(
        BING_SEARCH_V7_ENDPOINT,
        headers={"Ocp-Apim-Subscription-Key": subscription_key},
        params=params,
        timeout=DEFAULT_SEARCH_ENGINE_TIMEOUT,
    )
    if not response.ok:
        logger.error(f"{response.status_code} {response.text}")
        raise BingSearchValueError("status_code:" + response.status_code + "Search engine error.")
    json_content = response.json()
    try:
        contexts = json_content["webPages"]["value"][:REFERENCE_COUNT]
    except KeyError:
        logger.error(f"Error encountered: {json_content}")
        return []
    return contexts

class RAGSearchEngine():
    '''Used to build as an AI search engine'''
    def __init__(self):
        # 初始化AzureOpenAI客户端，使用环境变量中的Azure OAI端点和API密钥
        self.client = AzureOpenAI(azure_endpoint = os.getenv('AZURE_OAI_ENDPOINT'), 
                                  api_key = os.getenv('AZURE_OAI_KEY'),  
                                  api_version = os.getenv('API_VERSION'))
        self.model = "gpt-35-turbo"
        # 设置一个尝试次数和等待时间，然后用for循环尝试以避免因为输出延时导致报错
        self.try_times = 10
        self.wait_time = 0.5
        
    def query_function(self,
                       query:str,
                       model:str
                       ):# ChatCompletionchunk
        '''Search for a question on Bing and send the search results to llm, which in turn will answer it'''
        # model 3.5 don't support `stop` param
        # 如果模型为gpt-35-turbo或gpt-35-turbo-16k，则停止词为None
        # set self.modol as the model you choose 
        self.model = model
        if self.model == "gpt-35-turbo" or "gpt-35-turbo-16k":
            stop_words = None

        if query == "" or None:
            raise gr.Error(i18n("Please enter a question you would like to query."))

        try:
            contexts = search_with_bing(query)
        except TimeoutError :
            raise TimeoutError("Bing API request timed out.")
        
        # 构建系统提示，使用上下文中的片段
        system_prompt = _rag_query_text.format(
            context="\n\n".join(
                [f"[[citation:{i+1}]] {c['snippet']}" for i, c in enumerate(contexts)]
            )
        )

        # stream=True启用流式输出
        response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                max_tokens=1024,
                stop=stop_words,
                stream=True,
                temperature=0.9,
            )
        
        # 一个小 trick ，避免使用 while true 导致跑飞
        for _ in range(self.try_times):
            collected_reply = []
            collected_chunks = []
            # 然后实时返回
            try:
                for chunk in response:
                    # 获取回复
                    collected_chunks.append(chunk)
                    single_reply = chunk.choices[0].delta.content

                    # 逐个添加到合集中
                    collected_reply.append(single_reply)
                    # 使用join将所有组合起来
                    # clean None in collected_messages
                    collected_messages = [m for m in collected_reply if m is not None]
                    full_reply_content = ''.join([m for m in collected_messages])
                    # 返回
                    # print(full_reply_content)
                    yield full_reply_content
                break
            except:
                time.sleep(self.wait_time)
