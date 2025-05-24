"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations
from typing import Dict, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from langchain_community.llms import Ollama
import asyncio


class StudyState(TypedDict):
    question: str
    answer: str

llm = Ollama(model="llama3")




async def factual_node(state: StudyState) -> StudyState:
    prompt = f"Answer this factual study question concisely: {state['question']}"
    response = await llm.ainvoke(prompt)
    return {"question": state["question"], "answer": response}

async def conceptual_node(state: StudyState) -> StudyState:
    prompt = f"Explain this concept clearly for a student: {state['question']}"
    response = await llm.ainvoke(prompt)
    return {"question": state["question"], "answer": response}


def classify_question(state: StudyState) -> str:
    q = state["question"]
    if any(keyword in q.lower() for keyword in ["define", "what is", "who is"]):
        return "factual"
    else:
        return "conceptual"
    
def router(state: StudyState) -> StudyState:
    return state
    
builder = StateGraph(StudyState)

builder.add_node("factual_node", factual_node)
builder.add_node("conceptual_node", conceptual_node)
builder.add_node("router", router)


builder.add_conditional_edges("router", classify_question, {
    "factual": "factual_node",
    "conceptual": "conceptual_node"
})

builder.set_entry_point("router")
builder.add_edge("factual_node", END)
builder.add_edge("conceptual_node", END)

graph = builder.compile()
# Define the graph

