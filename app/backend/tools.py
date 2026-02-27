from dotenv import load_dotenv
load_dotenv()

import arxiv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from tavily import TavilyClient



def search_arxiv(query:str, max_results: int = 5):
    """
    search articles on arxiv
    """

    print(f"Searching on arxiv: {query}")

    client = arxiv.Client(delay_seconds=3.0,num_retries=3)

    search = arxiv.Search(query=query, max_results=max_results, sort_by = arxiv.SortCriterion.Relevance)

    results = []

    try:

        for result in client.results(search):
            results.append({
                "title": result.title,
                "link": result.pdf_url,
                "summary": result.summary,
                "published": result.published.strftime("%Y-%m-%d")
            })
    
    except Exception as e:
        print(f"Arxiv error: {e}")
        return [{"title": "ArXiv Access Error", "link": "", "summary": f"An error Occured: {str(e)}. Please wait a second and try again", "published": ""}]

    return results

def search_web(query:str):

    """
    General web search for technical blogs , documentation etc.
    """

    print(f"Searching on Web: {query}")

    search = DuckDuckGoSearchRun()
    return search.run(query)

def fetch_url_content(url: str):

    """
    go to the link and fetch the content as a text
    """

    print(f"Reading: {url}")

    loader = AsyncHtmlLoader([url])
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    if docs_transformed:
        return docs_transformed[0].page_content[0:4000]
    
    return "Content could not be read"

def search_web_tavily(query:str):

    """
    General web search using Tavily for technical blogs, documentation etc.
    """

    print(f"Searching on Web (Tavily): {query}")

    client = TavilyClient()
    response = client.search(query=query, max_results=5, search_depth="basic")

    results = []
    for result in response["results"]:
        results.append(f"{result['title']}: {result['url']}")

    return "\n".join(results) if results else "No results found"

def fetch_url_tavily(url: str):

    """
    Go to the link and extract the content using Tavily extract
    """

    print(f"Reading (Tavily): {url}")

    client = TavilyClient()
    response = client.extract(urls=[url])

    if response["results"]:
        content = response["results"][0]["raw_content"]
        return content[0:4000] if content else "Content could not be read"

    return "Content could not be read"

if __name__ == "__main__":

    try:

        papers = search_arxiv("LLM Agents")
        print(f"\nThe first article found: {papers[0]['title']}")
        print(f"summary: {papers[0]['summary'][:100]}...")

    except Exception as e:

        print(f"An error occured: {e}")