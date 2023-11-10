# GPT-Gradio-Agent  

[中文文档](README_zh-cn.md) | **English**

Build your own GPT Agent and manage your GPT-driven knowledge base with Azure OpenAI API (or OpenAI API) and Gradio! 

> If you like this project, please star for it, this is the greatest encouragement to me!

## Basic Chatbox Interface

This is the basic chatbox interface where you can have a conversation directly with GPT, let them play the role of an expert and answer your questions with system prompts, and manage your multiple conversations.

![chat界面 v0 7](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/dfbcd600-075c-4306-8a3b-b87a50290316)

## GPT-driven knowledge base

In this interface, you can **create and manage your own Knowledge Base** (CRUD) and **have GPT answer your questions in conjunction with **specified documents** (or **the entire Knowledge Base**), enabling RAG (Retrieval Augmented Generation)

Very efficient!

![v0 7 RAG 界面](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/903ef0ba-20f4-449e-ac31-654953b930ba)

# Get Started

## Git

0. Use `git clone https://github.com/Wannabeasmartguy/GPT-Gradio-Agent.git` to pull the codes；

Then open a **Command Prompt (CMD)** in the folder where the codes are stored, and use `pip install -r requirements.txt` to install the runtime environment.

1. Get [Azure OpenAI API Key](https://portal.azure.com/#home);

2. Rename `.env_example` to `.env`. Change the environment variable:  `OPENAI_API_KEY` and `OPENAI_API_BASE`;  
  > `OPENAI_API_KEY`：Azure OpenAI key;  
  > `OPENAI_API_BASE`：Access key provided by Azure OpenAI.

3. Enjoy it!  
  Use `python GPT-Gradio-Agent.py` in your terminal to run the the codes.You can see the URL in your terminal, and the default local URL is http://127.0.0.1:7860.

## Release package 

If you don't have Git installed, you can download the latest code from the [release](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/releases) page.

After extracting, follow **Step 1 and Step 2 above** to configure the environment, and finally **double-click `Run.bat`** to run the code.

> The default username is 'admin' and the password is '123456', which can be changed by yourself.The login function is turned off by default.

# Todo List

- [x] Use system-level prompts for role-playing

- [x] Support for context quantity control

- [x] Stream response

- [x] Detailed configuration of additional parameters

- [x] Choose models

- [x] Dialogue management

- [x] RAG(Retrieval Augmented Generation)

  - [x] Chat with single file
  
  - [x] Summarize the file
  
  - [x] Local knowledge base management
  
  - [x] Chat with whole knowledge base
  
    - [ ] List citation sources
  
  - [x] Estimated cost of embedding files

- [ ] Import and export chat history
  
  - [x] chat history export
  
  - [ ] chat history import
  
# About

[![GPT-Gradio-Agent](https://github-readme-stats-wannabeasmartguy.vercel.app/api?username=Wannabeasmartguy&show_icons=true&theme=vue)](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent)    
