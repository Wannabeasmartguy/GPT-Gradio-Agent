import gradio as gr
import openai

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