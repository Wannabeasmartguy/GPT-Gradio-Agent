import gradio as gr
import openai
import json
import os

def list_vali_check(my_list:list):
    '''List validity check, and return the first element'''
    if len(my_list) == 1:
        last_element = my_list[0]  # 列表只有一个元素，直接访问该元素
    else:
        last_element = my_list[-1]  # 列表长度大于1，使用索引[-1]访问最后一个元素
    return last_element

# ---------- Local storage of chat logs ----------
def json_to_list(json_data):
    '''Convert the json object to a list'''
    lst = json.loads(json_data)
    # Return the list
    return lst

def list_to_json_save(lst:list[list],filename='chat_cache.json'):
    '''Convert the list to a json object'''
    json_data = json.dumps(lst,indent=4, ensure_ascii=False)
    # Return the json object
    with open(filename, 'w') as file:
        file.write(json_data)

def init_chatbot():
    '''Load chat logs from json, if json does not exist then return None'''
    if os.path.exists('chat_cache.json'):
        with open('chat_cache.json', 'r', encoding='GB18030') as file:
            json_data = file.read()
            lst = json.loads(json_data)
            gr.Info("Initialization Local Success")
            return lst
    else:
        gr.Info("Initialization Local Success(no history)")
        return None

# ---------- Chat log export ----------
def format_messages(messages):
    formatted_messages = []

    # Iterate through the messages list
    for user_message, ai_message in messages:
        # Create a dictionary for the user message with role='user' and content=user_message
        user_dict = {"role": "user", "content": user_message}
        # Create a dictionary for the AI message with role='assistant' and content=ai_message
        ai_dict = {"role": "assistant", "content": ai_message}
        # Append the user and AI dictionaries to the formatted_messages list
        formatted_messages.append(user_dict)
        formatted_messages.append(ai_dict)

    # Return the formatted_messages list
    return formatted_messages

def sum_file_name(formatted_messages:list[dict]):
    response = openai.ChatCompletion.create(
            engine="gpt-35-turbo",
            messages=formatted_messages,
            stop=None
        )
    sum_file_result = response.choices[0].message.content
    return sum_file_result

def export_to_markdown(messages, ex_name):
    markdown_text = ""
    file_path="ouput\{chat_history}.md".format(chat_history=ex_name)

    for user_message, ai_message in messages:
        markdown_text += "## 来自你的消息\n\n"
        markdown_text += user_message + "\n\n"
        markdown_text += "## 来自 AI 的消息\n\n"
        markdown_text += ai_message + "\n\n"

    with open(file_path, "w", encoding="utf-8") as file:
        file.write(markdown_text)

    return gr.Info("聊天记录已保存在“Output”目录下。")

def get_all_conversation_names():
    '''
    获取 Json 文件中所有对话的名称，并返回一个列表用于初始化 Radio
    '''
    filename = 'chat_cache.json'

    # 检查 JSON 文件是否存在
    if os.path.isfile(filename):
        # 如果文件存在，则读取文件内容并解析为字典
        with open(filename, 'r', encoding='GBK') as file:
            data = json.load(file)
        
        # 获取所有对话名称并添加到列表中
        global history_list
        history_list = list(data.keys())
        return history_list
    else:
        print("JSON 文件不存在。")
        return ["New chat(1)"]

def get_last_conversation_content():
    '''
    用于获取 Json 文件中的最后一个对话的对话内容（最新保存的放在最后），用于初始化chatbot
    '''
    filename = 'chat_cache.json'

    # 检查 JSON 文件是否存在
    if os.path.isfile(filename):
        # 如果文件存在，则读取文件内容并解析为字典
        with open(filename, 'r', encoding='GBK') as file:
            data = json.load(file)
        
        # 获取最后一个对话的内容
        last_name = list(data.keys())[-1]
        last_content = data[last_name]
        return last_content
    else:
        print("JSON 文件不存在。")
        return []
    
def get_last_conversation_name():
    '''
    用于获取 Json 文件中的最后一个对话的对话名称（最新保存的放在最后），用于初始化chat_name。
    如果 `chat_cache.json` 不存在，则创建它。
    '''
    filename = 'chat_cache.json'

    # 检查 JSON 文件是否存在
    if os.path.isfile(filename):
        # 如果文件存在，则读取文件内容并解析为字典
        with open(filename, 'r', encoding='GBK') as file:
            data = json.load(file)
        
        # 获取最后一个对话的内容
        last_name = list(data.keys())[-1]
        return last_name
    else:
        print("JSON 文件不存在。")
        data = {
            "New chat(1)": []
        }
        # 将新的字典转换为 JSON 格式并写入文件
        with open(filename, 'w', encoding='GBK') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        last_name = list(data.keys())[-1]
        return last_name
    
def get_selected_conversation_content(name:str):
    '''
    用于获取 Json 文件中的指定对话名称的对话内容，用于更新chatbot
    '''
    filename = 'chat_cache.json'

    # 检查 JSON 文件是否存在
    if os.path.isfile(filename):
        # 如果文件存在，则读取文件内容并解析为字典
        with open(filename, 'r', encoding='GBK') as file:
            data = json.load(file)
        
        # 获取指定对话的内容
        selected_content = data[name]
        return selected_content
    else:
        print("JSON 文件不存在。")
        return []

def add_conversation_to_json(name:str, conversation:list[list]):
    '''
    用于添加对话时，保存当前的对话内容（May changed in future）和名称保存入json中，并创建一个新对话
    '''
    filename = 'chat_cache.json'

    # 检查 JSON 文件是否存在
    if os.path.isfile(filename):
        # 如果文件已存在，则读取文件内容并解析为字典
        with open(filename, 'r', encoding='GBK') as file:
            data = json.load(file)
    else:
        # 如果文件不存在，则创建一个新的字典
        data = {}

    # 将对话名称和对话内容添加到字典中
    data[name] = conversation

    # 创建一个空的新对话以刷新界面
    new = "New chat"

    # 检查是否存在相同的键
    if new in data:
        # 寻找可用的编号
        i = 1
        new_key = f"{new}({i})"
        while new_key in data:
            i += 1
            new_key = f"{new}({i})"
        data[new_key] = []
    else:
        data[new] = []

    # 将更新后的字典转换为 JSON 格式
    json_data = json.dumps(data, ensure_ascii=False, indent=4)

    # 将 JSON 数据写入文件
    with open(filename, 'w', encoding='GBK') as file:
        file.write(json_data)

def update_conversation_to_json(name:str, conversation:list[list]):
    '''
    对话内容更新时，将更新的内容保存入json中
    '''
    filename = 'chat_cache.json'

    # 检查 JSON 文件是否存在
    if os.path.isfile(filename):
        # 如果文件已存在，则读取文件内容并解析为字典
        with open(filename, 'r', encoding='GBK') as file:
            data = json.load(file)
    else:
        # 如果文件不存在，则创建一个新的字典
        data = {}

    # 将对话名称和对话内容添加到字典中
    data[name] = conversation

    # 将更新后的字典转换为 JSON 格式
    json_data = json.dumps(data, ensure_ascii=False, indent=4)

    # 将 JSON 数据写入文件
    with open(filename, 'w', encoding='GBK') as file:
        file.write(json_data)

def delete_conversation_from_json(name):
    filename = 'chat_cache.json'

    # 检查 JSON 文件是否存在
    if os.path.isfile(filename):
        # 如果文件存在，则读取文件内容并解析为字典
        with open(filename, 'r', encoding='GBK') as file:
            data = json.load(file)
        
        if len(data) == 1:
            raise gr.Error("This conversation is already the last one.（该对话已经是最后一个。）")

        # 检查要删除的对话名称是否存在于字典中
        if name in data:
            # 从字典中删除对应的键值对
            del data[name]
            print(f"对话 '{name}' 已成功删除。")
        else:
            print(f"对话 '{name}' 不存在。")
        
        # 将更新后的字典转换为 JSON 格式
        json_data = json.dumps(data, ensure_ascii=False, indent=4)

        # 将 JSON 数据写入文件
        with open(filename, 'w', encoding='GBK') as file:
            file.write(json_data)
    else:
        print("JSON 文件不存在。")

def modify_conversation_name(old_name, new_name):
    filename = 'chat_cache.json'

    # 检查 JSON 文件是否存在
    if os.path.isfile(filename):
        # 如果文件存在，则读取文件内容并解析为字典
        with open(filename, 'r', encoding='GBK') as file:
            data = json.load(file)
        
        # 检查要修改的对话名称是否存在于字典中
        if old_name in data:
            # 判断新名称是否与旧名称相同
            if new_name in data:
                # 相同，则提示修改失败
                gr.Info("Duplicate name! Please rename it!(重复的名称！请您重新命名！)")
                return gr.Textbox(value=old_name)
            else:
                # 不相同，则替换对话名称
                # 避免出现空名字
                if new_name == "":
                    gr.Info("The name can not be empty!(名称不能为空！)")
                    return gr.Textbox(value=old_name)
                data[new_name] = data.pop(old_name)
                print(f"对话名称 '{old_name}' 已成功修改为 '{new_name}'。")

                # 将更新后的字典转换为 JSON 格式
                json_data = json.dumps(data, ensure_ascii=False, indent=4)

                # 将 JSON 数据写入文件
                with open(filename, 'w', encoding='GBK') as file:
                    file.write(json_data)
                return gr.Textbox(value=new_name)
        else:
            print(f"对话名称 '{old_name}' 不存在。")

    else:
        print("JSON 文件不存在。")

def rewrite_his_in_list(chat_name:str):
    '''(WILL BE DELETED SOON)Updating the names of conversations in the Conversation Radio'''
    global history_list
    history_list[0] = chat_name
    return history_list

def insert_with_numbered_duplicates(lst:list, item:str):
   '''
   Checks if the item is already in the list. If it is, it adds a number to the end of the item to make it unique. The function then inserts the new item at the beginning of the list.
   
   Parameters:
   lst (list): The list to insert the item into
   item (str): The item to insert into the list
   
   Returns:
   None
   '''
   count = 0  
   new_item = item  

   while new_item in lst:  
       count += 1  
       new_item = f"{item}({count})"  

   lst.insert(0, new_item)

def rename_duplicates(#lst
                      ):
    if os.path.isfile('chat_cache.json'):
        global history_list
        lst = history_list

        count_dict = {}  # 创建一个字典用于记录元素出现的次数

        for i in range(len(lst)):
            item = lst[i]
            if item in count_dict:
                count = count_dict[item]
                count += 1
                lst[i] = f"{item}({count})"
                count_dict[item] = count
            else:
                count_dict[item] = 0
    else:
        pass