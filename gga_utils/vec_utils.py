import pandas as pd
import tabulate


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
    dicts = [{"documents": doc, "ids": id, "metadatas": meta} for doc, id, meta in tuples]

    return dicts

def dict_to_html(x:list[dict],file_name:str,advance_info:bool, small=True, api=False):
    info_list = []
    for info in x:
        if file_name in info['metadatas']['source']:
            df = pd.DataFrame(info.items(), columns=['Key', 'Value'])
            df.index = df.index + 1
            df.index.name = 'index'
            if api:
                res = tabulate.tabulate(df, headers='keys')
                doc_content = text_to_html(df.loc[1]['Value'])
                if advance_info:
                    info_list.append(res)
                info_list.append(doc_content)
            else:
                res = tabulate.tabulate(df, headers='keys', tablefmt='unsafehtml')
                doc_content = text_to_html(df.loc[1]['Value'])
                if advance_info:
                    info_list.append(res)
                info_list.append(doc_content)
    
    final_res = '\n\n'.join(info_list)
    if small:
        return "<small>" + final_res + "</small>"
    else:
        return final_res

def text_to_html(x, api=False):
    '''
    Encodes metadata in Chroma into HTML text for display in Gradio.
    
    Args:
        x: Metadata to be converted from ChromaDB
        api: Flag indicating if the function is called from an API
        
    Returns:
        str: HTML representation of the metadata
    '''
    x += "\n\n"
    if api:
        return x
    return """
<style>
      pre {
        overflow-x: auto;
        font-family: "微软雅黑", sans-serif;
        font-size: 14px;
        white-space: pre-wrap;
        white-space: -moz-pre-wrap;
        white-space: -pre-wrap;
        white-space: -o-pre-wrap;
        word-wrap: break-word;
      }
    </style>
<pre>
%s
</pre>
""" % x