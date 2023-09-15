import gradio as gr
import openai
import time
from utils import format_io

#load_dotenv()

AZURE_OAI_ENDPOINT = "Your Azure OpenAI endpoint"
AZURE_OAI_KEY = "Your Azure OpenAI key"
AZURE_OAI_MODEL = "gpt-35-turbo"

openai.api_type = "azure"                           # only Azure OpenAI needed
openai.api_base = AZURE_OAI_ENDPOINT                # only Azure OpenAI needed
openai.api_version = "2023-07-01-preview"
openai.api_key = AZURE_OAI_KEY

#gr.Chatbot.postprocess = format_io

def deliver(message:str,chat_history:list, chat_history_list:list,system:str,context_length:int, temperature:float,):

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

    response = openai.ChatCompletion.create(
        engine=AZURE_OAI_MODEL,
        messages=chat_history,
        temperature=temperature,
        max_tokens=400,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    reply = response.choices[0].message.content
        
    # GPT reply
    chat_input = {
        "role": "assistant",
        "content": reply
    }
    chat_history.append(chat_input)
    chat_history_list.append([message,reply])

    # Trim the context length first
    if len(chat_history) > context_length:
        chat_history = [chat_history[0]]+chat_history[1-context_length:]
        
    
    # Then return reply
    # Gradio's streaming output requires an explicit loop
    # result = []
    # for char in reply:
    #     result.append(char)
    #     str_result = ''.join(result)
    #     time.sleep(0.02)
    #     yield str_result,message,chat_history
    return chat_history_list,message,chat_history,chat_history_list

with gr.Blocks() as demo:
    gr.Markdown(
        '''
        # <center>GPT AGENT<center>
        <center>Use the agent make your work and life much more efficient.<center>
        '''
    )
    with gr.Row():
        with gr.Column(scale=1.8):
            chat_bot = gr.Chatbot()

        with gr.Column():
            
            usr_msg = gr.State()
            chat_his = gr.State([])
            chat_his_list = gr.State([])
            with gr.Row():
                #gr.Button("Send", callback=lambda: gr.set_value(message, gr.get_value(message) + " " + gr.get_value(chat_bot)))
                with gr.Column(width=20):
                    with gr.Accordion("Commom Setting"):
                        System_Prompt = gr.Textbox("You are a helpful AI.", label="System Prompt")
                        Context_length = gr.Slider(0, 32, value=4, step=1, label="Context length")
        
                with gr.Accordion("Additional Setting"):
                    
                    Temperature = gr.Slider(0, 1, value=0.7, step=0.01, label="Temperature")
    
        message = gr.Textbox(label="Input your prompt",
                             info="'Shift + Enter' to begin an new line.")
        with gr.Column(scale=0.1):
            send = gr.Button("Send",)
            clear = gr.ClearButton([message, chat_bot,chat_his,chat_his_list],scale=1,size="sm")
    # 将需要输入和输出的所有句柄进行合并
    input_param = [message, chat_his, chat_his_list, System_Prompt, Context_length, Temperature]
    output_param = [chat_bot, usr_msg, chat_his, chat_his_list]

    message.submit(deliver,input_param, output_param)
    send.click(deliver,input_param, output_param)

demo.queue().launch(debug=True)