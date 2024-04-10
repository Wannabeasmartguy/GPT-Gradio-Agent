'''Tools for Agents, e.g., web crawlers, etc.'''
from langchain_core.tools import tool


@tool
def web_crewler(url: str) -> str:
    """Useful for data extraction. Input should be a URL string."""
    from langchain_community.document_loaders import AsyncChromiumLoader,AsyncHtmlLoader 
    from langchain_community.document_transformers import Html2TextTransformer,BeautifulSoupTransformer  

    # Load HTML  
    # loader = AsyncChromiumLoader(url)  
    loader = AsyncHtmlLoader(url)
    html = loader.load()

    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(html)
    return docs_transformed[0].page_content


@tool
def do_not_need_tools(text: str) -> str:
    """If you don't need tools, just use this to return the input text."""
    return text