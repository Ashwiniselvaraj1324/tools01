"""
streamlit_app.py
A tiny Streamlit chat‑style app to answer *right‑now* questions with Google Gemini 1.5 Flash
and live web search (DuckDuckGo).  Uses LangChain’s ZERO_SHOT_REACT_DESCRIPTION agent.

🔑  IMPORTANT – replace the hard‑coded GEMINI_API_KEY below with **your** real key.
"""

import logging
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType

# ────────────────────────────────────────────
# 1. Minimal logging (silence the LangChain chatty bits)
logging.getLogger("langchain").setLevel(logging.ERROR)

# 2. Put your Gemini key HERE (hard‑coded as requested) 🔑
GEMINI_API_KEY = "AIzaSyCf7rxCiOtbLI5SnmjDRV9PS9WxWWk8uTI"

# 3. Build the LLM & search tool just once (cached)
@st.cache_resource(show_spinner=False)
def build_agent():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",     # fast, up‑to‑date reasoning
        google_api_key=GEMINI_API_KEY,
        temperature=0.4,
    )
    search_tool = DuckDuckGoSearchRun()
    return initialize_agent(
        tools=[search_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
    )

agent = build_agent()

# ────────────────────────────────────────────
# 4. Streamlit UI
st.set_page_config(page_title="Live Q&A 🤖🌐", page_icon="🌐")
st.title("🌐 Real‑Time Q&A with Gemini + DuckDuckGo 🔍")

# Helpful subtitle & hint
st.markdown(
    "Ask about **anything happening right now** – news events, fresh facts, "
    "or recent releases. I’ll combine Google Gemini’s reasoning with live "
    "DuckDuckGo results.  \n\n"
    "Examples:  \n• *“Who won yesterday’s IPL match?”*  \n"
    "• *“Latest iPhone 16 specs leak?”*"
)

user_query = st.text_input("💬 Your question", "")
ask_btn = st.button("Ask 🤔")

# ────────────────────────────────────────────
# 5. Run the agent & display the answer
if ask_btn or (user_query and not st.session_state.get("ran_once")):
    st.session_state["ran_once"] = True  # avoid auto‑rerun loops
    if not user_query.strip():
        st.warning("Please type a question first. ✍️")
    else:
        with st.spinner("Thinking…"):
            try:
                response = agent.run(user_query)
                st.success("Here’s what I found:")
                st.write(response)
            except Exception as err:
                st.error("😔 Sorry, something went wrong!")
                st.exception(err)  # shows stack trace only in dev / when expanded
