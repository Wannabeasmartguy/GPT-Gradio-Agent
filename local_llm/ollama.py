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


def process_api_params(is_enable:bool=False,
                       choose_list:list=["max_tokens","frequency_penalty"],
                       **kwargs):
    '''
    处理 Ollama API 传入参数
     OpenAI 的参数和 Ollama 的参数在设置时，合适的数值大小不同
    此外，某些参数的名称也不同，详见 https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values

    Args:
        is_enable (bool): 是否启用参数处理
        choose_list (list): 需要保留到 API 请求的参数列表
        kwargs (dict): 传入的参数
    '''
    if is_enable:
        # 对option_kwargs进行处理，使其符合API的要求
        num_ctx = kwargs.pop('max_tokens', None)
        if num_ctx is not None:
            kwargs['num_ctx'] = num_ctx
        
        repeat_penalty = kwargs.pop('frequency_penalty', None)
        if repeat_penalty is not None:
            kwargs['repeat_penalty'] = repeat_penalty

        kwargs = {k: v for k, v in kwargs.items() if k in choose_list}

        return kwargs
    

def send_ollama_chat_request(messages:list,
                      model:str="qwen:7b-chat", 
                      **option_kwargs
                      ):
    """
    发送聊天请求

    Args:
        model (str): 聊天模型
        messages (list): 聊天消息列表，每个消息包括角色和内容,与 OpenAI message 格式相同

    Yields:
        dict: 服务器返回的聊天结果
    """
    url:str="http://localhost:11434/api/chat"

    kwargs = process_api_params(is_enable=True, **option_kwargs)

    data = {
        "model": model,
        "messages": messages,
        "stream": False, # 流式输出则注释掉此行
        "options": kwargs
    }
    response = requests.post(url, json=data, stream=True)
    # yield from process_chat_response(response) # 流式输出则取消注释此行
    return dict(response.json())


def get_ollama_model_list(url:str = 'http://localhost:11434/api/tags'):
    """
    获取标签列表
    Args:
        url: Ollama 的API地址，默认为 http://localhost:11434/api/tags
    """
    response = requests.get(url)
    tags = response.json()

    # 获取所有模型的名称
    model_list = [model['name'] for model in tags['models']]
    return model_list