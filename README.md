# GPT-Gradio-Agent  

[中文文档](README_zh-cn.md) | **English**

Build your own GPT Agent and manage your GPT-driven knowledge base with Azure OpenAI API (or OpenAI API) and Gradio! 

> If you like this project, please star for it, this is the greatest encouragement to me!

## Basic Chatbox Interface

This is the basic Chatbox, where you can have a conversation with GPT and have them play the role of an expert to answer your questions through System Prompt.
![英文参数](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/e24645f6-ee92-4d2e-9565-805e21250546)

## GPT-driven knowledge base

In this interface, you can **create and manage your own knowledge base** (CRUD) and **have GPT answer your questions** in conjunction with **specified documents** (or the **entire knowledge base**). 

Very efficient!

![chatfile界面](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/b04c6ccc-8ed7-4d99-8831-fde912ea6fcd)

# Get Started
0. Use `git clone https://github.com/Wannabeasmartguy/GPT-Gradio-Agent.git` to pull the codes；

> If you don't have Git installed, you can download the latest codes on the Github **Tag** page.

Then open a **Command Prompt (CMD)** in the folder where the codes are stored, and use `pip install -r requirements.txt` to install the runtime environment.

1. Get [Azure OpenAI API Key](https://portal.azure.com/#home);

2. Rename `.env_example` to `.env`. Change the environment variable:  `OPENAI_API_KEY` and `OPENAI_API_BASE`;  
  > `OPENAI_API_KEY`：Azure OpenAI key;  
  > `OPENAI_API_BASE`：Access key provided by Azure OpenAI.

3. Enjoy it!  
  Use `python GPT-Gradio-Agent.py` in your terminal to run the the codes.You can see the URL in your terminal, and the default local URL is http://127.0.0.1:7860.
> The default username is 'admin' and the password is '123456', which can be changed by yourself.The login function is turned off by default, if you want to activate it, just change `inbrowser=False` to `inbrowser=True` and set your own account.

# Todo List

- [x] Use system-level prompts for role-playing

- [x] Support for context quantity control

- [x] Stream response

- [x] Detailed configuration of additional parameters

- [x] Choose models

- [ ] Chat with document

  - [x] Chat with single file
  
  - [x] Summarize the file
  
  - [x] Local knowledge base management
  
  - [x] Chat with whole knowledge base
  
    - [ ] List citation sources
  
  - [x] Estimated cost of embedding files

- [ ] Import and export chat history

# About

[![GPT-Gradio-Agent](https://github-readme-stats-wannabeasmartguy.vercel.app/api?username=Wannabeasmartguy&show_icons=true&theme=vue)](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent)    
