# GPT-Gradio-Agent
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
2. Rename '.env_example' to '.env'. Change the environment variable: `AZURE_OAI_MODEL`, `AZURE_OAI_KEY` and `AZURE_OAI_ENDPOINT`;  
  > `AZURE_OAI_MODEL`： deployed model's name, not the **model type**;  
  > `AZURE_OAI_KEY`：Azure OpenAI key;  
  > `AZURE_OAI_ENDPOINT`：Access key provided by Azure OpenAI.

> If you want to chat with file,please set **`OPENAI_API_KEY`,`OPENAI_API_BASE`,`OPENAI_API_VERSION` and `OPENAI_API_TYPE`**.
> Actually, **`AZURE_OAI_KEY` is same with `OPENAI_API_KEY`**.In the same way, **`AZURE_OAI_ENDPOINT` and `OPENAI_API_BASE` are the same**.
> *This method has to be used at the moment, and it will be optimized later.*

3. Enjoy it!
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

---

# GPT-Gradio-Agent
使用 Azure OpenAI 和 Gradio 创建自己的 GPT 智能助手
> 如果你喜欢这个项目，请你为它点上 star，这是对我最大的鼓励！

按“Chat with file”按钮，它会告诉你你想要的！
> "Send"直接与GPT聊天，"Chat with file"是与文件聊天。二者都在同一个界面展示，但**记忆不相通**。
![中文对话](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/9a48b9cc-f85a-4213-aa90-22276c6f14fc)

你还可以使用 "系统提示"（System Prompt），让它扮演您想要的角色，并设置参数。

## 快速开始
0. 使用`git clone https://github.com/Wannabeasmartguy/GPT-Gradio-Agent.git`拉取代码；
> 如果你没有安装 Git ，也可以只下载`GPT-Gradio-Agent.py`和`.env_example`。 
1. 获取[Azure OpenAI API Key](https://portal.azure.com/#home);
2. 将`.env_example`重命名为`.env`，修改环境变量`AZURE_OAI_MODEL`、`AZURE_OAI_KEY`和`AZURE_OAI_ENDPOINT`;
  > `AZURE_OAI_MODEL`：模型名称，写的是你自己**部署的模型名称**；  
  > `AZURE_OAI_KEY`：Azure OpenAI 的 key；  
  > `AZURE_OAI_ENDPOINT`：Azure OpenAI 的提供的“终结点”

> 如果要与文件对话，请设置 **`OPENAI_API_KEY`、`OPENAI_API_BASE`、`OPENAI_API_VERSION` 和`OPENAI_API_TYPE`**。
> 实际上，`AZURE_OAI_KEY`与`OPENAI_API_KEY`相同。同样，`AZURE_OAI_ENDPOINT`也与`OPENAI_API_BASE`相同。
> *目前不得不使用这种方法，后续将会对其进行优化*。

3. 尽情享受吧！
> 默认用户名为`admin`，密码为`123456`，可自行更改。

## 开发计划

- [x] （系统级提示）身份扮演

- [x] 添加上下文

- [x] 流式输出

- [x] 详细的参数配置

- [x] 模型选择

- [x] 支持与文件对话
  
  - [x] 单文件对话
  
  - [ ] 多文件对话
  
  - [ ] 对话文件管理

- [ ] 数据本地存储

- [ ] 聊天记录导入、导出

