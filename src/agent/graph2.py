import os
import tkinter as tk
from tkinter import filedialog
from io import BytesIO
from typing import TypedDict, List
from langchain.schema import Document
from langgraph.graph import StateGraph, END
import asyncio

class QAState(TypedDict):
    file_bytes: BytesIO
    file_name: str
    question: str
    documents: List[Document]
    answer: str


async def pick_file_node(state: QAState) -> QAState:
    root = tk.Tk()
    root.withdraw()  
    file_path = filedialog.askopenfilename(
        title="Select a file",
        filetypes=[("PDF files", "*.pdf"), ("Text files", "*.txt")]
    )
    if not file_path:
        raise ValueError("No file selected")

    with open(file_path, "rb") as f:
        file_bytes = BytesIO(f.read())

    return {
        **state,
        "file_name": file_path,
        "file_bytes": file_bytes,
    }


async def load_document_node(state: QAState) -> QAState:
    from langchain.document_loaders import PyPDFLoader, TextLoader
    import tempfile

    suffix = os.path.splitext(state["file_name"])[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(state["file_bytes"].read())
        tmp_path = tmp.name

    if suffix == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif suffix == ".txt":
        loader = TextLoader(tmp_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    docs = loader.load()
    os.unlink(tmp_path)
    return {**state, "documents": docs}

async def retrieval_qa_node(state: QAState) -> QAState:
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chains import RetrievalQA
    from langchain.llms import Ollama

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(state["documents"])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)

    retriever = vectorstore.as_retriever()
    llm = Ollama(model="llama3")
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    answer = qa.run(state["question"])
    return {**state, "answer": answer}




graph = StateGraph(QAState)
graph.add_node("pick_file", pick_file_node)
graph.add_node("load_document", load_document_node)
graph.add_node("retrieval_qa", retrieval_qa_node)


graph.set_entry_point("pick_file")
graph.add_edge("pick_file", "load_document")
graph.add_edge("load_document", "retrieval_qa")
graph.add_edge("retrieval_qa", END)

qa_graph = graph.compile()

'''
import asyncio

async def main():
    question = input("Enter your question: ")
    result = qa_graph.invoke({
        "file_bytes": BytesIO(),  # placeholder
        "file_name": "",
        "question": question,
        "documents": [],
        "answer": "",
    })
    print("\n✅ Answer:\n", result["answer"])

if __name__ == "__main__":
    asyncio.run(main())
'''