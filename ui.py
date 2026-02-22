import streamlit as st
import requests
import os

# Sets up the browser tab title, favicon, and centered layout for a cleaner UI
st.set_page_config(
    page_title="AI Academic Search",
    page_icon="🤖",
    layout="centered"
)

# We use an environment variable for the API URL to ensure the app works
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/ask")

st.title("Academic Search Assistant AI")


#Streamlit reruns the script on every interaction.
#We use st.session_state to persist chat history across these reruns.

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing chat history from the session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask a question (example: 'Search arxiv for Deep Learning')..."):
    
    # Display and store the user's message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty() #Create an empty placeholder to update the assistant's response dynamically
        full_response = ""
        
        with st.status("Agent is thinking...", expanded=True) as status:
            try:
                payload = {"input": prompt}
                response = requests.post(API_URL, json=payload, timeout=120)

                if response.status_code == 200: # Check if the backend responded successfully
                    data = response.json()
                    full_response = data.get("output", "")
                    
                    if not full_response: # Fallback for empty responses
                        full_response = "Backend reached, but output is empty."
                    
                    status.update(label="Answer Received!", state="complete", expanded=False)
                
                else:# Handle API-level errors
                    full_response = f"API Error: {response.status_code}"
                    status.update(label="API Error", state="error")

            except Exception as e: # Handle network or connection-level errors
                full_response = f"Connection Error: {str(e)}"
                status.update(label="Connection Error", state="error")

        if full_response:
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})