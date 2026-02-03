"""
RAG: retrieve chunks → prompt LLM → return answer + sources.
"""
from langchain_ollama import ChatOllama

from vectorstore import search

llm = ChatOllama(model="llama3.2", base_url="http://localhost:11434")

_PROMPT = """Use the following context from clinical protocol documents to answer the question. If the context does not contain relevant information, say so.

Context:
{context}

Question: {question}

Answer:"""


def query(question: str, k: int = 4) -> dict:
    """Retrieve relevant chunks, ask LLM, return answer and sources."""
    sources = search(question, k=k)
    context = "\n\n---\n\n".join(h["content"] for h in sources)
    prompt = _PROMPT.format(context=context, question=question)
    response = llm.invoke(prompt)
    return {
        "answer": response.content,
        "sources": [{"content": h["content"][:300], "metadata": h["metadata"]} for h in sources],
    }


out = query('Which LLM is used in this project?')
print(out['answer'])
print('--- sources ---')
for s in out['sources']:
    print(s['metadata'].get('source'), s['content'][:80])