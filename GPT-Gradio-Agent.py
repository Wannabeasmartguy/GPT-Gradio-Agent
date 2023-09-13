import os
#from dotenv import load_dotenv
import gradio as gr
import openai
import time

#load_dotenv()

AZURE_OAI_ENDPOINT = "Your Azure OpenAI endpoint"
AZURE_OAI_KEY = "Your Azure OpenAI key"
AZURE_OAI_MODEL = "gpt-35-turbo"

openai.api_type = "azure"                           # only Azure OpenAI needed
openai.api_base = AZURE_OAI_ENDPOINT                # only Azure OpenAI needed
openai.api_version = "2023-07-01-preview"
openai.api_key = AZURE_OAI_KEY


def deliver(message, history,system,context_length:int, temperature,):
    
    if "chat_history" not in globals():
        global chat_history
        chat_history = []

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

    # Trim the context length first
    if len(chat_history) > context_length:
        chat_history = [chat_history[0]]+chat_history[1-context_length:]
    
    #Then return reply
    return reply

def rst_memery():
    chat_history = []
    return chat_history

with gr.Blocks() as demo:
    gr.Markdown(
        '''
        # <center>GPT Assistance<center>
        <center>Use the agent make your work and life much more efficient.<center>
        '''
    )
    System_Prompt = gr.Textbox("You are a helpful AI.", label="System Prompt")
    Context_length = gr.Slider(0, 32, value=4, step=1, label="Context length",render=False)
    Temperature = gr.Slider(0, 1, value=0.7, step=0.01, label="Temperature",render=False)

    gr.ChatInterface(
        deliver, 
        additional_inputs=[System_Prompt, 
                            #clr_mem_btn,
                            Context_length, 
                            Temperature
                            ],
        theme="soft",
        submit_btn="Submit",
    )

demo.queue().launch(debug=True)
