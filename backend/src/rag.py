from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI
from operator import itemgetter
from decouple import config
from src.qdrant import vector_store, qdrant_search
from .openai_utils import stream_completion

model = ChatOpenAI(
    model_name="gpt-4o-mini",
    openai_api_key=config("OPENAI_API_KEY"),
    temperature=0,
)

prompt_template = """
Answer the question based on the context, in a concise manner, in markdown and using bullet points where applicable.

Context: {context}
Question: {question}
Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

retriever = vector_store.as_retriever()

def create_chain():
    chain = (
        {
            "context": retriever.with_config(top_k=4),
            "question": RunnablePassthrough(),
        }
        | RunnableParallel({
            "response": prompt | model,
            "context": itemgetter("context"),
            })
    )
    return chain

def get_answer_and_docs(question: str):
    chain = create_chain()
    response = chain.invoke(question)
    answer = response["response"].content
    context = response["context"]
    return {
        "answer": answer,
        "context": context
    }

async def async_get_answer_and_docs(question: str):
    docs = qdrant_search(query=question)
    docs_dict = [doc.payload for doc in docs]
    yield {
        "event_type": "on_retriever_end",
        "content": docs_dict
    }

    async for chunk in stream_completion(question, docs_dict):
        yield {
            "event_type": "on_chat_model_stream",
            "content": chunk
    }

    yield {
        "event_type": "done"
    }