import ollama
import os

import streamlit as st

import utils.logs as logs

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Load from env variable

if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API Key! Set OPENAI_API_KEY in your environment.")

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from llama_index.core.query_engine.retriever_query_engine import RetrieverQueryEngine

###################################
#
# Create Client
#
###################################


def create_client(host: str):
    """
    Creates a client for interacting with the Ollama API.

    Parameters:
        - host (str): The hostname or IP address of the Ollama server.

    Returns:
        - An instance of the Ollama client.

    Raises:
        - Exception: If there is an error creating the client.

    Notes:
        This function creates a client for interacting with the Ollama API using the ollama library. It takes a single parameter, host, which should be the hostname or IP address of the Ollama server. The function returns an instance of the Ollama client, or raises an exception if there is an error creating the client.
    """
    try:
        client = ollama.Client(host=host)
        logs.log.info("Ollama chat client created successfully")
        return client
    except Exception as err:
        logs.log.error(f"Failed to create Ollama client: {err}")
        return False


###################################
#
# Get Models
#
###################################

def get_models():
    try:
        chat_client = create_client(st.session_state["ollama_endpoint"])
        if not chat_client:
            logs.log.error("Ollama client could not be created.")
            return []

        # Get the list of models
        data = chat_client.list()

        # Debugging: Check response structure
        print(f"Raw API Response Type: {type(data)}")
        print(f"Raw API Response: {data}")

        # Ensure 'data' has the attribute 'models'
        if not hasattr(data, "models"):
            logs.log.error(f"Unexpected response format: {data}")
            return []

        # Extract model names from the response
        models = [model.model for model in data.models]  # Access the 'models' attribute

        st.session_state["ollama_models"] = models

        if models:
            logs.log.info(f"Ollama models loaded successfully: {models}")
        else:
            logs.log.warning("Ollama did not return any models. Make sure to download some!")

        return models

    except Exception as err:
        logs.log.error(f"Failed to retrieve Ollama model list: {err}")
        return []





###################################
#
# Create Ollama LLM instance
#
###################################


@st.cache_data(show_spinner=False)
def create_ollama_llm(model: str, base_url: str, system_prompt: str = None, request_timeout: int = 60) -> Ollama:
    """
    Create an instance of the Ollama language model.

    Parameters:
        - model (str): The name of the model to use for language processing.
        - base_url (str): The base URL for making API requests.
        - request_timeout (int, optional): The timeout for API requests in seconds. Defaults to 60.

    Returns:
        - llm: An instance of the Ollama language model with the specified configuration.
    """
    try:
        # Settings.llm = Ollama(model=model, base_url=base_url, system_prompt=system_prompt, request_timeout=request_timeout)
        Settings.llm = Ollama(model=model, base_url=base_url, request_timeout=request_timeout)
        logs.log.info("Ollama LLM instance created successfully")
        return Settings.llm
    except Exception as e:
        logs.log.error(f"Error creating Ollama language model: {e}")
        return None


###################################
#
# Chat (no context)
#
###################################


def chat(prompt: str):
    """
    Initiates a chat with the Ollama language model, enforcing step-by-step explanations.

    Parameters:
        - prompt (str): The starting prompt for the conversation.

    Yields:
        - str: Successive chunks of conversation from the Ollama model.
    """

    try:
        llm = create_ollama_llm(
            st.session_state["selected_model"],
            st.session_state["ollama_endpoint"],
        )

        # Step-by-step solution prompt
        step_by_step_prompt = f"""
        You are a math tutor providing structured, step-by-step solutions and explanations. Given the following query:

        1. Determine the Query Type:
        - Identify whether the question requires **solving a problem** (e.g., derivative, integral, proof) 
        or **explaining a concept** (e.g., theorem, definition, application).

       2. Logical Breakdown:
       - If solving a problem, provide **clear, step-by-step calculations** with justifications.
      - If explaining a concept, provide a **structured explanation** with definitions, key properties, examples, 
     and real-world applications.

       3. Application of Theorems/Formulas:
       - If solving a problem, apply the necessary **theorems, formulas, or identities**.
       - If explaining a concept, reference relevant **mathematical principles** and their significance.

        4. Final Answer & Summary:
       - If solving a problem, present the **final result clearly**, with verification if necessary.
        - If explaining a concept, provide a **concise summary** of key takeaways.

Query: {prompt}
"""


        stream = llm.stream_complete(step_by_step_prompt)
        for chunk in stream:
            yield chunk.delta
    except Exception as err:
        logs.log.error(f"Ollama chat stream error: {err}")
        return



###################################
#
# Document Chat (with context)
#
###################################

def context_chat(prompt: str, query_engine: RetrieverQueryEngine):
    """
    Initiates a chat with context using the Llama-Index query_engine and ensures step-by-step explanations.

    Parameters:
        - prompt (str): The starting prompt for the conversation.
        - query_engine (RetrieverQueryEngine): The Llama-Index query engine to use for retrieving answers.

    Yields:
        - str: Successive chunks of conversation from the Llama-Index model with context.
    """
    try:
        # Step-by-step solution prompt
        step_by_step_prompt = f"""
        You are a math tutor providing structured, step-by-step solutions and explanations. Given the following query:

        1. Determine the Query Type:
        - Identify whether the question requires **solving a problem** (e.g., derivative, integral, proof) 
         or **explaining a concept** (e.g., theorem, definition, application).

        2. Logical Breakdown:
        - If solving a problem, provide **clear, step-by-step calculations** with justifications.
        - If explaining a concept, provide a **structured explanation** with definitions, key properties, examples, 
     and real-world applications.

        3. Application of Theorems/Formulas:
        - If solving a problem, apply the necessary **theorems, formulas, or identities**.
        - If explaining a concept, reference relevant **mathematical principles** and their significance.

        4. Final Answer & Summary:
        - If solving a problem, present the **final result clearly**, with verification if necessary.
        - If explaining a concept, provide a **concise summary** of key takeaways.

       Query: {prompt}
"""

        # Query the engine with the structured prompt
        stream = query_engine.query(step_by_step_prompt)
        for text in stream.response_gen:
            yield str(text)
    except Exception as err:
        logs.log.error(f"Ollama chat stream error: {err}")
        return


