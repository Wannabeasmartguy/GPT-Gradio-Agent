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

search_quote_icon = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Image and Text Layout</title>
<style>
  body {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100vh;
    margin: 0;
    background-color: #f0f0f0;
  }
  .image-container {
    display: flex;
    align-items: center;
  }
  .image {
    width: 100px; /* Adjust the width as needed */
    height: auto; /* Adjust the height as needed */
  }
  .text {
    color: #3181ED;
    margin-left: 20px; /* Adjust the spacing between image and text */
  }
</style>
</head>
<body>
<div class="image-container">
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-book-text "><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"></path><path d="M8 7h6"></path><path d="M8 11h8"></path></svg>
  <div class="text">
    <h1>Sources</h1>
  </div>
</div>
</body>
</html>
"""

search_Answer_icon = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Image and Text Layout</title>
<style>
  body {
    display: flex;
    align-items: center;
    justify-content: center;
    height: 100vh;
    margin: 0;
    background-color: #f0f0f0;
  }
  .image-container {
    display: flex;
    align-items: center;
  }
  .image {
    width: 100px; /* Adjust the width as needed */
    height: auto; /* Adjust the height as needed */
  }
  .text {
    color: #3181ED;
    margin-left: 20px; /* Adjust the spacing between image and text */
  }
</style>
</head>
<body>
<div class="image-container">
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-book-open-text "><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path><path d="M6 8h2"></path><path d="M6 12h2"></path><path d="M16 8h2"></path><path d="M16 12h2"></path></svg>
  <div class="text">
    <h1>Answer</h1>
  </div>
</div>
</body>
</html>
"""

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
    "[end]",
    "\nReferences:\n",
    "\nSources:\n",
    "End.",
]

BING_MKT = "en-US"
BING_SEARCH_V7_ENDPOINT = os.getenv("BING_SEARCH_URL")
subscription_key = os.getenv("BING_SUBSCRIPTION_KEY")
DEFAULT_SEARCH_ENGINE_TIMEOUT = 5
REFERENCE_COUNT = 6

class BingSearchValueError(Exception):
    pass

def search_with_bing(query: str, subscription_key: str = os.getenv("BING_SUBSCRIPTION_KEY")):
    """
    Search with bing and return the contexts.
    """
    params = {"q": query, "mkt": BING_MKT}
    
    # 先检查 BING_SEARCH_V7_ENDPOINT 和 subscription_key 是否为空
    if not BING_SEARCH_V7_ENDPOINT or not subscription_key:
        # raise BingSearchValueError("BING_SEARCH_V7_ENDPOINT and subscription_key cannot be empty.")
        raise gr.Error(i18n("BING_SUBSCRIPTION_KEY and BING_SEARCH_URL cannot be empty."))
    
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

def dict_to_html_card(item:list[dict]):
    """
    将搜索结果字典转换为一个HTML卡片的字符串。
    
    参数:
    item -- 字典，包含了要转换为HTML卡片的信息。
    
    返回值:
    string -- 表示HTML卡片的字符串。
    """
    # 定义HTML卡片的模板
    card_template = '''
    <a href="{url}" class="card">
        
        <div class="card-content">
            <div class="title">{name}</div>
            <div class="snippet">{snippet}</div>
            <div class="url">{url}</div>
        </div>
    </a>
    '''
    # 使用字典中的值填充模板
    card = card_template.format(
        url=item['url'],
        # thumbnailUrl=item['thumbnailUrl'],
        name=item['name'],
        snippet=item['snippet'],
        dateLastCrawled=item['dateLastCrawled']
    )
    return card

def list_to_html_page(items:str):
    """
    将一个列表转换为一个HTML页面的字符串，每个元素都被转换为一个HTML卡片。
    
    参数:
    items -- 列表，包含了要转换为HTML卡片的字典。
    
    返回值:
    string -- 表示HTML页面的字符串。
    """
    # 定义HTML页面的开头和结尾
    html_start = '''
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>搜索结果卡片</title>
    <style>
        .card-container {
            display: flex;
            flex-wrap: wrap;
        }
        .card {
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            margin: 10px;
            background-color: #f9f9f9;
            flex: 1 0 25%;
            transition: box-shadow 0.3s;
            width: 150px; /* 或者使用百分比宽度，如 width: 100%; */
            box-sizing: border-box; /* 确保宽度包括边框和内边距 */
            text-decoration: none; /* 移除链接的下划线 */
            color: inherit;
        }
        .card:hover {
            box-shadow: 0 0 11px rgba(33,33,33,.2); 
        }
        .card img {
            width: 80px;
            height: 80px;
            object-fit: cover;
            border-radius: 4px;
        }
        .card .title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .card .snippet {
            color: #666;
            font-size: 14px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .card .url {
            font-size: 12px;
            color: #999;
            text-overflow: ellipsis;
            overflow: hidden;
        }
        .title_txt {
            font-size: 18px;
            font-weight: bold;
            color: #3b82f6;
        }
    </style>
    </head>
    <body>
    <div class="card-container">
    '''
    html_end = '</div></body></html>'
    
    # 给结合后的卡片组一个标题
    html_title = '''
    <div class="title_txt">
        <div class="title">Source</div>
    </div>
    '''

    # 将列表中的每个字典转换为一个HTML卡片
    cards = [dict_to_html_card(item) for item in items]
    
    # 将所有卡片连接成一个字符串
    html_body = ''.join(cards)
    
    # 将HTML页面的开头、主体和结尾连接成一个完整的HTML页面
    html_page = html_title + html_start + html_body + html_end
    
    return html_page

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
        self.contexts = []
        self.stop_words = stop_words
        
    def query_function(self,
                       query:str,
                       model:str='gpt-35-turbo',
                       ):# ChatCompletionchunk
        '''Search for a question on Bing and send the search results to llm, which in turn will answer it'''
        # model 3.5 don't support `stop` param
        # 如果模型为gpt-35-turbo或gpt-35-turbo-16k，则停止词为None
        # set self.modol as the model you choose 
        self.model = model
        if self.model == "gpt-35-turbo" or "gpt-35-turbo-16k":
            stop_words = None
        else:
            stop_words = self.stop_words

        if query == "" or None:
            raise gr.Error(i18n("Please enter a question you would like to query."))

        try:
            contexts = search_with_bing(query)
            self.contexts = contexts
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

                    # 用html装饰一下
                    search_answer_formated = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Search result</title>
                        <style>
                            .colored-text {{
                                font-size: 18px;
                                color: #58606E;
                            }}
                        </style>
                    </head>
                    <body>
                        <p class="colored-text">{str(full_reply_content)}</p>
                    </body>
                    </html>
                    """
                    yield search_answer_formated
                break
            except:
                time.sleep(self.wait_time)

    def gen_html_page(self):
        '''返回搜索结果转换得到的html'''
        html_page = list_to_html_page(self.contexts)
        return html_page
    
    def get_contexts(self):
        '''返回搜索引擎的搜索结果'''
        return self.contexts