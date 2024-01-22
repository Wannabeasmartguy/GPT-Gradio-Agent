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

**v0.9.0 New**: You can now view not only the file directory of the knowledge base, but also the specific content in the knowledge base, and understand the specific chunks.

![knowledge base info](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/883315f0-b2f0-4c39-acc9-6f4104fe9609)

Click the checkbox to view the specific information of the block:

![detail of knowledge base info](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/d8e94529-ad52-4d28-becf-bddcce94d5d6)

It's important to note that when you load a knowledge base, it will automatically load out its specific information.

Also, it's a stand-alone module, which means you can use it to view Knowledge Base A when you open Knowledge Base B at the same time!

## Dall-E-3 Image Generator

If you have access to the Azure OpenAI API, it's a waste not to try the Dall-E-3 model to generate images!

![Dall-E-3](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/6b8c7e7c-8c75-41a0-b0ce-46f69bb7a9ef)

# Get Started

## Git

0. Use `git clone https://github.com/Wannabeasmartguy/GPT-Gradio-Agent.git` to pull the codes；

Then open a **Command Prompt (CMD)** in the folder where the codes are stored, and use `pip install -r requirements.txt` to install the runtime environment.

1. Get [Azure OpenAI API Key](https://portal.azure.com/#home);

> The model **deployment name** **MUST** be the **same** as the **model name**!
> For example, when deploying `'gpt-35-turbo'`, the deployment name of the model should also be `'gpt-35-turbo'`.

> If you want to use the **latest model `gpt-4-1106-preview`** (which is call gpt-4-turbo in OpenAI Conference) ,you should **use `'gpt-4-turbo-pr'` as the deployment name**.

2. Copy `.env_example` and rename it to `.env`, change the environment variables:
    Starting from **v0.9.0**, there is a change in **environmental variables**: There is a conflict between Langchain's environment variable settings for OpenAI and Azure OpenAI, and it is not possible to use Azure Openai properly when both variables are set at the same time, which means that OpenAI cannot be supported at the same time from this version (I will look for a compatible method in the future), so you need to refer to the following environment variable settings. This also means that OpenAI cannot be supported from this release onwards (I will look for a way to make it compatible in the future), so you need to set the environment variables as follows.
  > `AZURE_OAI_KEY`: api key for Azure OpenAI;
  > `AZURE_OAI_ENDPOINT`: Azure OpenAI's provided "endpoint";
  > `API_VERSION`: the version of the API used by Azure OpenAI; **NOTE**: use `2023-12-01-preview` if you want to use Dall-E-3, `2023-09-15-preview` and earlier will be deprecated on April 2, 2024;
  > `API_TYPE`: indicates to use Azure OpenAI instead of OpenAI.

3. Enjoy it!  
  Use `python GPT-Gradio-Agent.py` in your terminal to run the the codes.You can see the URL in your terminal, and the default local URL is http://127.0.0.1:7860.

## Release package 

If you don't have Git installed, you can download the latest code from the [release](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/releases) page.

After extracting, follow **Step 1 and Step 2 above** to configure the environment, and finally **double-click `Run.bat`** to run the code.

> The default username is 'admin' and the password is '123456', which can be changed by yourself.The login function is turned off by default.

# Todo List

- [x] Adaptive language adaptation (Chinese, English supported)

- [x] Use system-level prompts for role-playing

- [x] Support for context quantity control

- [x] Stream response

- [x] Detailed configuration of additional parameters

- [x] Choose models
  
  - [x] Support gpt-4-turbo(which is called `gpt-4-1106-preview` in Azure)

  - [ ] Image input with gpt-4-turbo support (requires additional deployment of Azure's Computer Vision resources)
  
  - [x] Generating images using Dall-E 3

- [x] Dialogue management

- [x] **RAG(Retrieval Augmented Generation)**

  - [x] **Chat with single file**
  
  - [x] Summarize the file
  
  - [x] Local knowledge base management
  
  - [x] **Chat with whole knowledge base**
  
    - [x] List citation sources
  
  - [x] Estimated cost of embedding files

- [x] **Web requester(By prompts)**

- [ ] Import and export chat history
  
  - [x] chat history export
  
  - [ ] chat history import
  
# About

[![GPT-Gradio-Agent](https://github-readme-stats-wannabeasmartguy.vercel.app/api?username=Wannabeasmartguy&show_icons=true&theme=vue)](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent)    
