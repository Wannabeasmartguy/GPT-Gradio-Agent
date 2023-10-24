# GPT-Gradio-Agent、

**中文文档** | [English](README.md)

使用 Azure OpenAI API（或 OpenAI API）和 Gradio 构建自己的 GPT 助手并管理 GPT 驱动的知识库！
> 如果你喜欢这个项目，请你为它点上 star，这是对我最大的鼓励！

## 基本聊天框界面

这是基本的聊天框界面，你可以在其中直接与GPT进行对话，并通过系统提示让他们扮演专家的角色并回答您的问题。

![英文参数](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/e24645f6-ee92-4d2e-9565-805e21250546)

## GPT 驱动的知识库

在此界面中，您可以**创建和管理自己的知识库**（CRUD），并**让 GPT 结合**指定的文档（或**整个知识库**）回答您的问题。

> 不用一行行地找真是太高效了！

![chatfile界面](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/b04c6ccc-8ed7-4d99-8831-fde912ea6fcd)

# 快速开始

## GIT

0. 使用`git clone https://github.com/Wannabeasmartguy/GPT-Gradio-Agent.git`拉取代码；
然后在**命令提示符 (CMD)**中打开你的运行环境，使用 `pip install -r requirements.txt` 安装运行依赖。

1. 获取[Azure OpenAI API Key](https://portal.azure.com/#home);

2. 复制 `.env_example`并重命名为`.env`，修改环境变量`OPENAI_API_KEY`和`OPENAI_API_BASE`;  
  > `OPENAI_API_KEY`：Azure OpenAI 的 api key；  
  > `OPENAI_API_BASE`：Azure OpenAI 的提供的“终结点”

1. 尽情享受吧！  
   在终端输入`python GPT-Gradio-Agent.py` 以运行代码。在终端内你可以看到本地 URL，它一般是 http://127.0.0.1:7860。

## Release 下载便携包

如果你没有安装 Git ，可以在 [release](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/releases) 页面下载最新代码。

解压后，按照上面的**步骤 1 和步骤 2** 配置环境，最后**双击`run.bat`**运行代码。

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