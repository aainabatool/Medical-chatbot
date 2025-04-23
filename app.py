import os
import streamlit as st
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper
from langchain_groq import ChatGroq
from langchain.tools.wolfram_alpha import WolframAlphaQueryRun
from langchain.utilities.wolfram_alpha import WolframAlphaAPIWrapper
import nest_asyncio

# Apply asyncio patch
nest_asyncio.apply()

# Set your GROQ and WolframAlpha API keys here or through environment variables
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["WOLFRAM_ALPHA_APPID"] = st.secrets["WOLFRAM_ALPHA_APPID"]

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192"
)

# Initialize tools
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
wolfram = WolframAlphaQueryRun(api_wrapper=WolframAlphaAPIWrapper())

tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for searching Wikipedia articles"
    ),
    Tool(
        name="WolframAlpha",
        func=wolfram.run,
        description="Useful for math, science, and factual queries"
    )
]

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Streamlit UI
st.title("Medical Chatbot with LangChain & Groq")

query = st.text_input("Ask a medical or factual question:")
if query:
    with st.spinner("Thinking..."):
        response = agent.run(query)
        st.write("**Response:**")
        st.success(response)
