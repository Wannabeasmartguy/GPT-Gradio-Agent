**中文文档** | [English](README.md)

使用 Azure OpenAI API（或 OpenAI API）和 Gradio 构建自己的 GPT 助手并管理 GPT 驱动的知识库！
> 如果你喜欢这个项目，请你为它点上 star，这是对我最大的鼓励！

## 基本聊天框界面

这是基本的聊天框界面，你可以在其中直接与GPT进行对话，通过系统提示让他们扮演专家的角色并回答您的问题,并管理你的多个对话。

![chat界面 v0 7](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/f0ec31dc-6cdf-42db-9f8c-4dceae9dabca)

## GPT 驱动的知识库

在此界面中，您可以**创建和管理自己的知识库**（CRUD），并**让 GPT 结合**指定的文档（或**整个知识库**）回答您的问题，实现 RAG （检索增强生成）。

> 不用一行行地在文件里找内容真是太高效了！

![v0 7 RAG 界面](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/705c8f58-d46b-487a-b4d5-9cc38397397f)

# 快速开始

## GIT

0. 使用`git clone https://github.com/Wannabeasmartguy/GPT-Gradio-Agent.git`拉取代码；
然后在**命令提示符 (CMD)**中打开你的运行环境，使用 `pip install -r requirements.txt` 安装运行依赖。

1. 获取[Azure OpenAI API Key](https://portal.azure.com/#home);

> 模型的**部署名称**，**一定要**和模型的原名称相同！
> 比如部署 `'gpt-35-turbo'` 时，模型的部署名称也要填 `'gpt-35-turbo'`

2. 复制 `.env_example`并重命名为`.env`，修改环境变量`OPENAI_API_KEY`和`OPENAI_API_BASE`;  
  > `OPENAI_API_KEY`：Azure OpenAI 的 api key；  
  > `OPENAI_API_BASE`：Azure OpenAI 的提供的“终结点”

1. 尽情享受吧！  
   在终端输入`python GPT-Gradio-Agent.py` 以运行代码。在终端内你可以看到本地 URL，它一般是 http://127.0.0.1:7860。

## Release 下载便携包

如果你没有安装 Git ，可以在 [release](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/releases) 页面下载最新代码。

解压后，按照上面 **GIT** 的**步骤 1 和步骤 2** 配置环境，最后**双击`run.bat`**运行代码。

## 开发计划

- [x] （系统级提示）身份扮演

- [x] 添加上下文

- [x] 流式输出

- [x] 详细的参数配置

- [x] 模型选择

- [x] 对话管理 

- [x] 检索增强生成（RAG）
  
  - [x] 单文件对话
  
  - [x] 文件全文总结

  - [x] 知识库本地管理

  - [x] 知识库全局检索与对话
    
    - [ ] 显示引用来源
  
  - [x] 预估嵌入文件的费用

- [ ] 聊天记录导入、导出

  - [x] 聊天记录导出
  
  - [ ] 聊天记录导入

# 关于
[![GPT-Gradio-Agent](https://github-readme-stats-wannabeasmartguy.vercel.app/api?username=Wannabeasmartguy&show_icons=true&theme=vue)](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent)    
