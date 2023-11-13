sum_wechat_gzh = """在 >>> 和 <<< 之间是网页的返回的HTML内容。
网页内容是一篇文章。
我希望你能扮演我的文字秘书、文字改进员的角色，对这篇文章进行总结概要。
你将精读全文，使用简洁干练的语言，将文本中的主要内容提炼出来。

RULES：
1. 必须完整引用涉及准确的事实数据和人物原话，不要省略；
2. 必须完整引用原文已经分点列举的重要内容，包括“优点”、“存在的问题和风险”、“具体的举措”等。

>>> {requests_result} <<<
请使用如下的Markdown格式结构返回总结内容，并输出在 >> 和 << 之间的完整代码块：

>>
```markdown
# 标题
xxxxxxx

# 作者
<公众号名称 or "佚名">

# 日期
xxxx年xx月xx日

# 内容

## 文章主题
xxxxxx

## 涉及对象
对象a、对象b......

## 具体内容概述 

### <章节1名称>
xxxxxx

### <章节2名称>
xxxxxx

......

```
<<

Extracted:"""

sina_test = """在 >>> 和 <<< 之间是网页的返回的HTML内容。
网页是新浪财经A股上市公司的公司简介。
请抽取参数请求的信息。

>>> {requests_result} <<<
请使用如下的JSON格式返回数据
{{
  "company_name":"a",
  "company_english_name":"b",
  "issue_price":"c",
  "date_of_establishment":"d",
  "registered_capital":"e",
  "office_address":"f",
  "Company_profile":"g"

}}
Extracted:"""