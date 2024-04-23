from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper

import json
import requests
from bs4 import BeautifulSoup

RESULTS_PER_QUESTION = 3

ddg_search = DuckDuckGoSearchAPIWrapper()


def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]


template = """Summarize the following question based on the context:

Question: {question}

Context: 

{context}"""

prompt = ChatPromptTemplate.from_template(template)

SUMMARY_TEMPLATE = """{text} 
-----------
Using the above text, answer in short the following question: 
> {question}
-----------
if the question cannot be answered using the text, imply summarize the text. Include all factual information, numbers, stats etc if available."""  # noqa: E501
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)


def scrape_text(url: str):
    # send GET to request the webpage
    try:
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator=" ", strip=True)
            return page_text
        else:
            return f"Failed to retreivethe webpage: Status Code: {response.status_code}"
    except Exception as e:
        print(e)
        return "Failed to retreive the webpage: {e}"


url = "https://blog.langchain.dev/announcing-langsmith/"

scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary=RunnablePassthrough.assign(text=lambda x: scrape_text(x["url"])[:10000])
    | SUMMARY_PROMPT
    | ChatOpenAI(model="gpt-3.5-turbo-1106")
    | StrOutputParser()
) | (lambda x: f"URL: {x['url']}\n\nSUMMARY: {x['summary']}")


# build a chain that runs ddg search to find urls, then build a list of dicts that hold the url and original question
# pass that into the scrape and summarize to then scrape the urls that were found by the search and then
# feed into LLM for summarization
web_search_chain = (
    RunnablePassthrough.assign(urls=lambda x: web_search(x["question"]))
    | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]])
    | scrape_and_summarize_chain.map()
)

# now build sub searches to answer the initial question, in this case 3 queries that will
# be returned as a list in json
SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 google search queries to search online that form an "
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format:"
            '["query 1", "query 2", "query 3"].',
        ),
    ]
)

search_question_chain = (
    SEARCH_PROMPT | ChatOpenAI(temperature=0) | StrOutputParser() | json.loads
)

full_research_chain = (
    search_question_chain
    | (lambda x: [{"question": q} for q in x])
    | web_search_chain.map()
)

WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."  # noqa: E501


# Report prompts from https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/master/prompts.py
RESEARCH_REPORT_TEMPLATE = """Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career."""  # noqa: E501

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)


def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)


chain = (
    RunnablePassthrough.assign(
        research_summary=full_research_chain | collapse_list_of_lists
    )
    | prompt
    | ChatOpenAI(model="gpt-3.5-turbo-1106")
    | StrOutputParser()
)


# # now take the question generator chain, get its queries and map to the web search chain
# chain = (
#     search_question_chain
#     | (lambda x: [{"question": q} for q in x])
#     | web_search_chain.map()
# )
# x = chain.invoke(
#     {
#         "question": "What is the difference between langsmith and langchain?",
#     }
# )


#!/usr/bin/env python
from fastapi import FastAPI
from langserve import add_routes


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    chain,
    path="/research-assistant",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
