import streamlit as st
from streamlit_chat import message

from backend import query_db
from utils import stream_data
from agents.tavily_search_agent import tavily_search_agent


# Initialize session state for storing messages
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("JooAI")
st.write("")
st.write("")

message("Greetings! How may I assist you today?", key="welcome_message")

# Display chat messages from session state
for idx, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg["is_user"], key=f"message_{idx}")

# Input field for user queries
query = st.chat_input("Ask your queries here")

if query:
    # Append the user message to the chat
    st.session_state.messages.append({"content": query, "is_user": True})
    message(query, is_user=True, key=f"user_message_{len(st.session_state.messages)}")

    # AI response
    with st.spinner("Looking it up for you, just a moment"):
        response = query_db(query=query)
        
        if "Sorry, I don't know the answer. I need to search Google for the answer." in response:
            rephrased_query = f"{query} - Millennium Balti Restaurant"
            response = tavily_search_agent(input=rephrased_query)
            response_text = response
        else:
            # Ensure stream_data result is converted to a single string
            response_text = response
        
        # Append the AI response to the chat
        st.session_state.messages.append({"content": response_text, "is_user": False})
        message(response_text, is_user=False, key=f"ai_message_{len(st.session_state.messages)}")


