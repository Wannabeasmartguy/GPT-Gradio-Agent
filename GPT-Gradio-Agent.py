import gradio as gr
import openai
import time
#from utils import format_io

#load_dotenv()

AZURE_OAI_ENDPOINT = "Your Azure OpenAI endpoint"
AZURE_OAI_KEY = "Your Azure OpenAI key"
AZURE_OAI_MODEL = "gpt-35-turbo"

openai.api_type = "azure"                           # only Azure OpenAI needed
openai.api_base = AZURE_OAI_ENDPOINT                # only Azure OpenAI needed
openai.api_version = "2023-07-01-preview"
openai.api_key = AZURE_OAI_KEY

#gr.Chatbot.postprocess = format_io

def deliver(message:str,
            chat_history:list, 
            chat_history_list:list,
            system:str,
            context_length:int, 
            temperature:float,
            max_tokens:int,
            top_p:float,
            frequency_penalty:float,
            presence_penalty:float):

    # System Prompt and User Prompt
    if system:
        system_input = {
            "role": "system",
            "content": system
        }
        if chat_history == []:
            chat_history.append(system_input)
        else:
            chat_history[0] = system_input
 
    user_input = {
        "role": "user",
        "content": message
    }

    chat_history.append(user_input)

    if context_length == 0:
        # If context_length == 0,clean up chat_history
        response = openai.ChatCompletion.create(
            engine=AZURE_OAI_MODEL,
            messages=[system_input,user_input],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=None
        )
    else:
        response = openai.ChatCompletion.create(
            engine=AZURE_OAI_MODEL,
            messages=chat_history,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=None
        )
    reply = response.choices[0].message.content
        
    # GPT reply
    chat_input = {
        "role": "assistant",
        "content": reply
    }
    chat_history.append(chat_input)
    chat_history_list.append([message,None])

    # Trim the context length first
    if (len(chat_history)-1 > context_length) and len(chat_history)>3:
        chat_history = [chat_history[0]]+chat_history[-context_length:]
    
    return chat_history_list,message,chat_history

def stream(history_list:list,chat_history:list[dict]):
    bot_message = chat_history[-1]['content']
    history_list[-1][1] = ""
    for character in bot_message:
        history_list[-1][1] += character
        time.sleep(0.02)
        yield history_list

with gr.Blocks() as demo:
    gr.Markdown(
        '''
        # <center>GPT AGENT<center>
        <center>Use the agent make your work and life much more efficient.<center>
        '''
    )
    usr_msg = gr.State()
    chat_his = gr.State([])
    with gr.Row():
        with gr.Column(scale=1.8):
            chat_bot = gr.Chatbot(height=500,
                                  show_copy_button=True,
                                  bubble_full_width=False)
            message = gr.Textbox(label="Input your prompt",
                                        info="'Shift + Enter' to begin an new line. Press 'Enter' can also send your Prompt to the LLM.")
            with gr.Row(scale=0.1):
                clear = gr.ClearButton([message, chat_bot,chat_his],scale=1,size="sm")
                send = gr.Button("Send",scale=2)

        with gr.Column():
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("Commom Setting"):
                        System_Prompt = gr.Textbox("You are a helpful AI.", label="System Prompt",
                                                   info="'Shift + Enter' to begin an new line.")
                        Context_length = gr.Slider(0, 32, value=4, step=1, label="Context length",
                                                   info="每次请求携带的历史消息数")                    
        
                    with gr.Accordion("Additional Setting"):
                        max_tokens = gr.Slider(0, 4096, value=400, step=1, label="max_tokens",
                                               info="携带上下文交互的最大 token 数")
                        Temperature = gr.Slider(0, 2, value=0.5, step=0.1, label="Temperature",
                                                info="随机性：值越大，回复越随机")
                        top_p = gr.Slider(0, 1, value=1, step=0.1, label="top_p",
                                          info="核采样：与随机性类似，但不要与随机性一起修改")
                        frequency_penalty = gr.Slider(-2, 2, value=0, step=0.1, label="frequency_penalty",
                                                      info="频率惩罚度：值越大，越不容易出现重复字词")
                        presence_penalty = gr.Slider(-2, 2, value=0, step=0.1, label="frequency_penalty",
                                                      info="话题新鲜度：值越大，越可能扩展到新的话题")
        

    # Merge all handles that require input and output.
    input_param = [message, chat_his, chat_bot, System_Prompt, 
                   Context_length, Temperature,max_tokens,top_p,frequency_penalty,
                   presence_penalty]
    output_param = [chat_bot, usr_msg, chat_his]

    message.submit(deliver,input_param, output_param, queue=False).then(stream,[chat_bot,chat_his],chat_bot)
    send.click(deliver,input_param, output_param, queue=False).then(stream,[chat_bot,chat_his],chat_bot)

demo.queue().launch(inbrowser=False,debug=True,auth=[("admin","123456")],
                    auth_message="欢迎使用 GPT-Gradio-Agent ,请输入用户名和密码")