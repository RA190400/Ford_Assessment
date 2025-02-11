import streamlit as st
import re
from utils.ollama import chat, context_chat

# Function to process and format LaTeX responses
def format_latex_response(response_generator):
    response_text = "".join(response_generator)  # Convert generator to string

    # Convert LaTeX delimiters to Streamlit-compatible formatting
    response_text = response_text.replace("\\(", "$").replace("\\)", "$")  # Inline math
    response_text = response_text.replace("\\[", "$$").replace("\\]", "$$")  # Block math

    # Handle environments like equation, align, gather, etc.
    response_text = re.sub(r'\\begin{equation}(.*?)\\end{equation}', r'$$\1$$', response_text, flags=re.DOTALL)
    response_text = re.sub(r'\\begin{align}(.*?)\\end{align}', r'$$\1$$', response_text, flags=re.DOTALL)
    response_text = re.sub(r'\\begin{alignat}(.*?)\\end{alignat}', r'$$\1$$', response_text, flags=re.DOTALL)
    response_text = re.sub(r'\\begin{gather}(.*?)\\end{gather}', r'$$\1$$', response_text, flags=re.DOTALL)
    response_text = re.sub(r'\\begin{CD}(.*?)\\end{CD}', r'$$\1$$', response_text, flags=re.DOTALL)

    return response_text

def chatbox():
    if prompt := st.chat_input("How can I help?"):
        # Prevent submission if Ollama endpoint is not set
        if not st.session_state.get("query_engine"):
            st.warning("Please confirm settings and upload files before proceeding.")
            st.stop()

        # Add the user input to messages state
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response from llama-index or query engine
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response_generator = context_chat(prompt=prompt, query_engine=st.session_state["query_engine"])
                
                response_text = ""  # Initialize empty response
                response_container = st.empty()  # Create dynamic container for updates
                
                # Stream response incrementally
                for chunk in response_generator:
                    response_text += chunk  # Append each chunk
                    formatted_text = format_latex_response(response_text)  # Format incrementally
                    response_container.markdown(formatted_text, unsafe_allow_html=True)  # Update UI dynamically

        # Final formatted response
        formatted_response = format_latex_response(response_text)

        # Ensure last update is properly formatted
        response_container.markdown(formatted_response, unsafe_allow_html=True)

        # Add the final response to messages state
        st.session_state["messages"].append({"role": "assistant", "content": formatted_response})
