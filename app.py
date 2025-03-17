import streamlit as st
import time
import json
import os
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("⚠️ GOOGLE_API_KEY is missing! Please set it in the .env file.")

# ✅ Configure API Key (No need for ADC)
genai.configure(api_key=api_key)
datasci_chatbot = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=api_key)

# ✅ Chat Memory Handling
CHAT_STORAGE = "chat_logs"
os.makedirs(CHAT_STORAGE, exist_ok=True)

def load_chat_log(user_id):
    """Retrieve past conversations."""
    try:
        with open(f"{CHAT_STORAGE}/{user_id}.json", "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_chat_log(user_id):
    """Save conversations to file."""
    with open(f"{CHAT_STORAGE}/{user_id}.json", "w") as file:
        json.dump(st.session_state.conversation_history, file, indent=4)

def clear_chat_log(user_id):
    """Delete chat history."""
    try:
        os.remove(f"{CHAT_STORAGE}/{user_id}.json")
        st.session_state.conversation_history = []
        st.success("✅ Chat history erased.")
    except FileNotFoundError:
        st.warning("⚠️ No saved chat history to delete.")

def download_chat_log(user_id):
    """Download conversation as text file."""
    chat_data = load_chat_log(user_id)
    if chat_data:
        chat_text = "\n".join([f"User: {msg['user']}\nAI: {msg['ai']}\n" for msg in chat_data])
        st.download_button("⬇ Download Chat History", data=chat_text, file_name=f"{user_id}_chat_history.txt", mime="text/plain")
    else:
        st.warning("⚠️ No chat history available.")

# ✅ Streamlit UI Configuration
st.set_page_config(page_title="AI Data Science Tutor", page_icon="🧠", layout="centered")

# ✅ User Authentication
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🤖 Data Science Tutor")
    user_id = st.text_input("Enter your username:")
    if st.button("Start Chat"):
        if not user_id:
            st.error("⚠️ Please provide a username.")
        else:
            st.session_state.user = user_id
            st.session_state.authenticated = True
            st.session_state.conversation_history = load_chat_log(user_id)
            st.success("✅ Chat loaded!" if st.session_state.conversation_history else "🔄 New session started.")
            time.sleep(1)
            st.rerun()
    st.stop()

# ✅ Sidebar Controls
user_id = st.session_state.user
with st.sidebar:
    st.write(f"### 👋 Hello, {user_id.title()}!")
    st.write("""
#### Welcome to the AI-Powered Data Science Tutor! 🎓
🤖 I can help with:
- 🔹 Machine Learning & AI
- 🔹 Python for Data Science
- 🔹 Data Visualization
- 🔹 Deep Learning Concepts
    """)
    if st.button("🗑 Clear Chat History"):
        clear_chat_log(user_id)
    download_chat_log(user_id)

# ✅ Display Past Messages
if not st.session_state.conversation_history:
    st.chat_message("assistant").write("💡 Tip: Ask me anything about **data science**!")
else:
    for message in st.session_state.conversation_history:
        with st.chat_message("user"):
            st.markdown(f"**👨‍💻 You:** {message['user']}")
        with st.chat_message("assistant"):
            st.markdown(f"**🤖 Tutor:** {message['ai']}")

# ✅ AI Prompt & Chain Setup
chat_prompt = ChatPromptTemplate(
   messages=[
    ("system", """
    You are an expert **Data Science Tutor**. Your goal is to provide **to-the-point** and **structured** answers.
    
    **Guidelines for answering:**
    - ✅ Keep responses **short, clear, and structured** (point-wise format).
    - 📌 **Use examples** only when necessary.
    - 🐍 **Include Python code** **only if** it directly helps in understanding.
    - 🚫 If a question is **not related to Data Science**, reply with:
      *"I specialize in Data Science. Please ask a relevant question."*
    """),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{user_query}")
]

)

output_parser = StrOutputParser()
history_loader = RunnableLambda(load_chat_log)
processing_chain = RunnablePassthrough.assign(history=history_loader) | chat_prompt | datasci_chatbot | output_parser

# ✅ User Input Handling
user_query = st.chat_input("💬 Ask me about Data Science...")

if user_query:
    with st.chat_message("user"):
        st.markdown(f"**👨‍💻 You:** {user_query}")

    with st.chat_message("assistant"):
        ai_response_placeholder = st.empty()
        response_text = ""
        
        with st.spinner("🤖 Thinking..."):
            try:
                ai_reply = processing_chain.invoke({"user_query": user_query})
                for word in ai_reply.split():
                    response_text += word + " "
                    time.sleep(0.02)  # Simulate typing effect
                    ai_response_placeholder.markdown(response_text)
                
                # Store conversation
                st.session_state.conversation_history.append({"user": user_query, "ai": response_text})
                save_chat_log(user_id)
                st.rerun()
            except Exception as err:
                st.error(f"⚠️ Error: {err}")
