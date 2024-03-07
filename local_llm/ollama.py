import requests
import json

def process_chat_response(response):
    """
    处理聊天API的流式输出

    Args:
        response (requests.Response): 聊天API的响应对象

    Yields:
        dict: 每个聊天消息的结果
    """
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            yield data

def send_chat_request(messages:list,
                      model:str="qwen:7b-chat", 
                      url:str="http://localhost:11434/api/chat"):
    """
    发送聊天请求

    Args:
        url (str): 请求的URL
        model (str): 聊天模型
        messages (list): 聊天消息列表，每个消息包括角色和内容,与 OpenAI message 格式相同

    Yields:
        dict: 服务器返回的聊天结果
    """
    data = {
        "model": model,
        "messages": messages,
        "stream": False # 流式输出则注释掉此行
    }
    response = requests.post(url, json=data, stream=True)
    # yield from process_chat_response(response) # 流式输出则取消注释此行
    return dict(response.json())
