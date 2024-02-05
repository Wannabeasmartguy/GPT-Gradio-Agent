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

**v0.9.0 新增**：你现在不仅可以查看知识库的文件目录，还可以查看知识库内的具体内容，并且了解具体的分块情况。

![knowledge base info](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/adec4553-8815-419b-9b72-445bd6176c63)

点击勾选框，就可以查看分块的具体信息：

![detail of knowledge base info](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/1c29dcd7-aa0b-4c87-820d-2fdee57e4144)

需要指出的是，当你加载了一个知识库后，它就会自动加载出它的具体信息。

并且，它是一个独立的模块，这也就意味着，你可以同时在打开知识库 A 时，使用该模块查看知识库 B ！

## Dall-E-3 图片生成器

如果你有 Azure OpenAI API 的访问权限，那么不尝试一下 Dall-E-3 模型来生成图片真是太浪费了！基于其非常强大的图片生成能力，它可能会成为你的得力助手。

![Dall-E-3](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/6b8c7e7c-8c75-41a0-b0ce-46f69bb7a9ef)

## RAG 搜索

通过利用搜索引擎（目前支持Bing）的广泛知识和 LLM 的卓越功能，您可以获得精确的答案和广泛的参考资料。它的灵感来自 [search_with_lepton](https://github.com/leptonai/search_with_lepton)。

![search with gpt4](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/assets/107250451/89cdcb56-82c1-4b2c-a2ef-60a63bc8dfe0)

# 快速开始

## GIT

0. 使用`git clone https://github.com/Wannabeasmartguy/GPT-Gradio-Agent.git`拉取代码；
然后在**命令提示符 (CMD)**中打开你的运行环境，使用 `pip install -r requirements.txt` 安装运行依赖。

1. 获取[Azure OpenAI API Key](https://portal.azure.com/#home);

> 模型的**部署名称**，**一定要**和模型的原名称相同！
> 比如部署 `'gpt-35-turbo'` 时，模型的部署名称也要填 `'gpt-35-turbo'`
> 如果你要用 Dall-E-3 模型，那么部署名称请使用 `'dall-e-3'`

2. 复制 `.env_example`并重命名为`.env`，修改环境变量：  
    从 **v0.9.0** 开始，**环境变量与之前存在变化**：Langchain 对 OpenAI 和 Azure OpenAI 的环境变量设置存在冲突，无法在同时设置二者的变量时正常使用 Azure Openai ，这也意味着从该版本开始，暂时无法同时支持 OpenAI（我会在未来寻找兼容的方法），因此需要参照以下对环境变量进行设置。
  > `AZURE_OAI_KEY`：Azure OpenAI 的 api key；  
  > `AZURE_OAI_ENDPOINT`：Azure OpenAI 的提供的“终结点”；
  > `API_VERSION`：Azure OpenAI 使用的 API 版本；**注意**：如果你想使用 Dall-E-3，请使用 `2023-12-01-preview` ，`2023-09-15-preview` 及更早的版本将于 2024年 4 月 2 日被废弃；
  > `API_TYPE`：表示使用 Azure OpenAI 而非 OpenAI。

  > 如果你要使用 RAG Search，请设置 `BING_SUBSCRIPTION_KEY` ，你可以在 Azure 中获得相关的免费资源，每月可以有 1000 次的免费调用机会。

3. 尽情享受吧！  
   在终端输入`python GPT-Gradio-Agent.py` 以运行代码。在终端内你可以看到本地 URL，它一般是 http://127.0.0.1:7860。

## Release 下载便携包

如果你没有安装 Git ，可以在 [release](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent/releases) 页面下载最新代码。

解压后，按照上面 **GIT** 的**步骤 1 和步骤 2** 配置环境，最后**双击`run.bat`**运行代码。

## 开发计划

- [x] 自适应语言适配（已支持中文、英语）

- [x] （系统级提示）身份扮演

- [x] 添加上下文

- [x] 流式输出

- [x] 详细的参数配置

- [x] 模型选择

  - [x] 支持 gpt-4-turbo（ Azure 中名称为`gpt-4-1106-preview`）
  
  - [ ] 支持 gpt-4-turbo 的图片输入（需要额外部署 Azure 的 Computer Vision 资源）
  
  - [x] 支持使用 Dall-E 3 生成图片

- [x] 对话管理 

- [x] **检索增强生成（RAG）**
  
  - [x] **单文件对话**
  
  - [x] 文件全文总结

  - [x] 知识库本地管理

  - [x] **知识库全局检索与对话**
    
    - [x] 显示引用来源
  
  - [x] 预估嵌入文件的费用

- [x] RAG 强化搜索引擎搜索
  
  - [x] 支持 Bing 搜索引擎

- [x] 网页内容检索及总结（通过提示词）

- [ ] 聊天记录导入、导出

  - [x] 聊天记录导出
  
  - [ ] 聊天记录导入

# 关于
[![GPT-Gradio-Agent](https://github-readme-stats-wannabeasmartguy.vercel.app/api?username=Wannabeasmartguy&show_icons=true&theme=vue)](https://github.com/Wannabeasmartguy/GPT-Gradio-Agent)    
