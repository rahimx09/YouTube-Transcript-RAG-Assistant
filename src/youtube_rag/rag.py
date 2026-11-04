"""Retrieve transcript context and answer questions with Gemini."""

import logging
import os

from youtube_rag.indexing import get_vector_store
from youtube_rag.settings import CHAT_MODEL, RETRIEVAL_COUNT

LOGGER = logging.getLogger(__name__)


def answer_question(question: str) -> str:
    """Answer a question using only the most relevant indexed transcript chunks."""
    from dotenv import load_dotenv
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import (
        RunnableLambda,
        RunnableParallel,
        RunnablePassthrough,
    )
    from langchain_google_genai import ChatGoogleGenerativeAI

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is required to generate answers. "
            "Add it to your .env file or environment."
        )

    LOGGER.info("Retrieving the %d most relevant transcript chunks.", RETRIEVAL_COUNT)
    retriever = get_vector_store().as_retriever(
        search_type="similarity",
        search_kwargs={"k": RETRIEVAL_COUNT},
    )
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer only from the provided transcript
context. If the context is insufficient, say that you don't know.

Transcript context:
{context}

Question: {question}"""
    )
    model = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=api_key,
    )
    chain = (
        RunnableParallel(
            {
                "context": RunnableLambda(
                    lambda query: "\n\n".join(
                        document.page_content for document in retriever.invoke(query)
                    )
                ),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | model
        | StrOutputParser()
    )

    LOGGER.info("Generating an answer from the retrieved transcript context.")
    return chain.invoke(question)
