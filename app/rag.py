from functools import lru_cache

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from app.config import get_settings

settings = get_settings()


@lru_cache
def get_chain() -> RetrievalQA:
    """Build or fetch a cached Retrieval-Augmented-Generation chain."""
    store = Chroma(
        persist_directory=settings.chroma_dir,
        embedding_function=OpenAIEmbeddings(
            api_key=settings.openai_api_key
        ),
    )
    retriever = store.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model=settings.model_name,
        temperature=0.0,
        openai_api_key=settings.openai_api_key,
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
