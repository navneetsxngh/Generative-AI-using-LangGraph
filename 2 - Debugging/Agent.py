import os
from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass
from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool

from langchain_nvidia_ai_endpoints import ChatNVIDIA

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGSMITH_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "ReAct Agent"


@dataclass
class State:
    messages: Annotated[list[AnyMessage], add_messages]

llm = ChatNVIDIA(
  model="openai/gpt-oss-20b",
  api_key= os.getenv('NVIDIA_API_KEY'), 
  temperature=0.7,
  top_p=1,
  max_tokens=4096,
)

def make_default_graph():
    graph_workflow = StateGraph(State)

    def call_model(state:State):
        messages = [SystemMessage(content="You are a helpful AI assistant.")] + state.messages
        response = llm.invoke(messages)
        return {"messages": [response]}
    
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("agent", END)   
    graph_workflow.add_edge(START, "agent")

    agent = graph_workflow.compile()
    return agent

agent = make_default_graph()