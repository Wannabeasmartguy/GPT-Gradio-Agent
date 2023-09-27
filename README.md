# GPT-Gradio-Agent

[中文文档](README_zh.md)

Build your own GPT Agent with  Azure OpenAI API and Gradio. 

> If you like this project, please star for it, this is the greatest encouragement to me!

Press the "Chat with file" button, it will tell you what you want!
> The "Send" button let you to chat with GPT **directly**，and the "Chat with file" button let you to **chat with file**。Both are presented in the same interface, but **memory is not connected**.
![英文对话](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/8ac16117-6b48-42f9-a521-7d9702ba9c9b)

You can also use "System Prompt" to let it play a role of what you want, and set the parameters.
![英文参数](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/e24645f6-ee92-4d2e-9565-805e21250546)

## Get Started
0. Use `git clone https://github.com/Wannabeasmartguy/GPT-Gradio-Agent.git` to pull the codes；
> If you don't have Git installed, you can also just download the `GPT-Gradio-Agent.py` and `.env_example`.

1. Get [Azure OpenAI API Key](https://portal.azure.com/#home);
![image](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/a6801676-1a5a-4b4d-8843-8237f04f2ee9)
> It is recommended to **create a new subscription**.
> In the new subscription, the **deploy name** of the model **must be the same as the model name**.

2. Rename `.env_example` to `.env`. Change the environment variable:  `OPENAI_API_KEY` and `OPENAI_API_BASE`;  
  > `OPENAI_API_KEY`：Azure OpenAI key;  
  > `OPENAI_API_BASE`：Access key provided by Azure OpenAI.

3. Enjoy it!  
  Use `python GPT-Gradio-Agent.py` in your terminal to run the codes.You can see the URL in your terminal, and the default local URL is http://127.0.0.1:7860.
> The default username is 'admin' and the password is '123456', which can be changed by yourself.

## Todo List

- [x] Use system-level prompts for role-playing

- [x] Support for context quantity control

- [x] Stream response

- [x] Detailed configuration of additional parameters

- [x] Choose models

- [x] Chat with documents

  - [x] Chat with single file
  
  - [ ] Chat with multiple files
  
  - [ ] Dialogue files management

- [ ] Local storage of data

- [ ] Import and export chat history
