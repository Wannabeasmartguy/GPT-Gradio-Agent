from langchain.prompts import PromptTemplate
from langchain.chains import LLMRequestsChain, LLMChain
from langchain.chat_models import AzureChatOpenAI
from openai.error import InvalidRequestError
import gradio as gr

def sum_chain(model:str,
              url:str,
              chat_bot:list[list]):
    '''
    构建一个总结网页内容（文章）的 LLMRequestsChain ，非 Agent
    '''

    llm = AzureChatOpenAI(model=model,
                    openai_api_type="azure",
                    deployment_name=model, # <----------设置选择模型的时候修改这里
                    temperature=0)

    template = """在 >>> 和 <<< 之间是网页的返回的HTML内容。
    网页内容是一篇文章。
    我希望你能扮演我的文字秘书、文字改进员的角色，对这篇文章进行总结概要。
    你将精读全文，使用简洁干练的语言，将文本中的主要内容提炼出来。

    RULES：
    1. 必须完整引用涉及准确的事实数据和人物原话，不要省略；
    2. 必须完整引用原文已经分点列举的重要内容，包括“优点”、“存在的问题和风险”、“具体的举措”等。

    >>> {requests_result} <<<
    请使用如下的Markdown格式结构返回总结内容
    
    # 标题
    xxxxxxx

    # 作者
    <公众号名称 or "佚名">

    # 日期
    xxxx年xx月xx日

    # 内容
    
    ## 文章主题
    xxxxxx

    ## 涉及对象
    对象a、对象b......

    ## 具体内容概述 
    
    ### <章节1名称>
    xxxxxx

    ### <章节2名称>
    xxxxxx

    ......
    
    Extracted:"""

    prompt = PromptTemplate(
        input_variables=["requests_result"],
        template=template
    )

    chain = LLMRequestsChain(llm_chain=LLMChain(llm=llm, prompt=prompt))

    inputs = {
    "url": url
    }

    try:
        response = chain(inputs)
        # print(response['output'])
        chat_bot.append([f"请为我总结这篇文章:{url}",response['output']])
        return chat_bot
    except InvalidRequestError:
        raise gr.Error("待总结的文章长度已经超过了模型能够承受的最大长度！请选择支持更长上下文的模型！")    