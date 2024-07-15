import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(base_url="http://localhost:11434/v1", model="llama3")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Set an initial system message if chat history is empty
if not st.session_state.chat_history:
    system_message = SystemMessage(content="You are a helpful AI assistant.")
    st.session_state.chat_history.append(system_message)

# Streamlit app layout
st.title("Chatbot with Streamlit")

# Display chat history
for i, message in enumerate(st.session_state.chat_history):
    if isinstance(message, HumanMessage):
        st.write(f"You: {message.content}")
    elif isinstance(message, AIMessage):
        st.write(f"AI: {message.content}")
    elif isinstance(message, SystemMessage):
        st.write(f"System: {message.content}")

# Create new input box for each new message
if 'input_key' not in st.session_state:
    st.session_state.input_key = 0

for i in range(st.session_state.input_key):
    query = st.text_input(f"You (message {i+1}):", key=f"input_{i}")
    if st.button(f"Send {i+1}", key=f"button_{i}"):
        if query:
            st.session_state.chat_history.append(HumanMessage(content=query))  # Add user message

            # Get AI response using history
            result = model.invoke(st.session_state.chat_history)
            response = result.content
            st.session_state.chat_history.append(AIMessage(content=response))  # Add AI message

            # Increment input_key to create a new input box
            st.session_state.input_key += 1
            st.experimental_rerun()

# Always have one input box available for the next message
query = st.text_input("You:", key=f"input_{st.session_state.input_key}")
if st.button("Send", key=f"button_{st.session_state.input_key}"):
    if query:
        st.session_state.chat_history.append(HumanMessage(content=query))  # Add user message

        # Get AI response using history
        result = model.invoke(st.session_state.chat_history)
        response = result.content
        st.session_state.chat_history.append(AIMessage(content=response))  # Add AI message

        # Increment input_key to create a new input box
        st.session_state.input_key += 1
        st.experimental_rerun()

st.write("---- Message History ----")
st.write(st.session_state.chat_history)