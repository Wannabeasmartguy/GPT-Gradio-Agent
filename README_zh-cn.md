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

  然后使用在命令行中输入 `pip install -r requirements.txt` 来安装运行环境。

1. 获取[Azure OpenAI API Key](https://portal.azure.com/#home);

2. 复制 `.env_example`并重命名为`.env`，修改环境变量`OPENAI_API_KEY`和`OPENAI_API_BASE`;  
  > `OPENAI_API_KEY`：Azure OpenAI 的 api key；  
  > `OPENAI_API_BASE`：Azure OpenAI 的提供的“终结点”

3. 尽情享受吧！  
   在终端输入`python GPT-Gradio-Agent.py` 以运行代码。在终端内你可以看到本地 URL，它一般是 http://127.0.0.1:7860。
> 默认用户名为`admin`，密码为`123456`，可自行更改。

## 开发计划

- [x] （系统级提示）身份扮演

- [x] 添加上下文

- [x] 流式输出

- [x] 详细的参数配置

- [x] 模型选择

- [x] 支持与文件对话
  
  - [x] 单文件对话
  
  - [x] 文件全文总结

  - [x] 知识库本地管理

  - [x] 知识库全局检索与对话
    
    - [ ] 显示引用来源
  
  - [x] 预估嵌入文件的费用

- [ ] 聊天记录导入、导出

# 关于
[![GPT-Gradio-Agent](https://github-readme-stats-wannabeasmartguy.vercel.app/api?username=Wannabeasmartguy&show_icons=true&theme=vue)](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent)    
