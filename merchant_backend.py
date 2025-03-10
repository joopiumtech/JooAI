import os
import ast
import openai
import sounddevice as sd
import numpy as np
import wave
# import pygame


from agents.tavily_search_agent import tavily_search
from utils import fetch_restaurant_name, initialize_db, insert_data_to_redis, retrieve_data_from_redis, verify_password
from langgraph.prebuilt import create_react_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
from dotenv import load_dotenv
load_dotenv(override=True)


# Initialize the LLM model
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_retries=1,
    api_key=os.environ.get("OPENAI_API_KEY"),
    streaming=True
)

    

def get_business_reference_data(query: str):
    # Initialize pinecone
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    pc = Pinecone(api_key=pinecone_api_key)
    index_name = os.environ.get("INDEX_NAME")
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=3072,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=os.environ.get("PINECONE_ENVIRONMENT")),
        )

    index = pc.Index(index_name)

    # Initialize embedding model
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.environ.get('OPENAI_API_KEY')
    )

    vector_store = PineconeVectorStore(index=index, embedding=embedding_model)

    results = vector_store.similarity_search(
        query,
        k=5,
        namespace="marketing_plan"
    )
    for res in results:
        return res.page_content


# Record user voice input
def record_audio(filename="input_audio/input.wav", duration=5, samplerate=16000):
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
    sd.wait()
    
    # Save the recording
    with wave.open(filename, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(samplerate)
        f.writeframes(audio_data.tobytes())


# Convert voice to text
def speech_to_text(filename="input_audio/input.wav"):
    with open(filename, "rb") as audio_file:
        response = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return response.text


# def text_to_speech(text, filename="response.mp3"):
#     """
#     Converts text to speech using OpenAI TTS and plays it.
#     """
#     response = openai.audio.speech.create(
#         model="tts-1",
#         voice="alloy",  # Available voices: alloy, echo, fable, onyx, nova, shimmer
#         input=text
#     )
#     with open(filename, "wb") as f:
#         f.write(response.content)

#     pygame.mixer.init()
#     pygame.mixer.music.load(filename)
#     pygame.mixer.music.play()
#     while pygame.mixer.music.get_busy():
#         continue


def query_db_for_merchant(query: str = None, audio_query: bool = False):
    """
    Authenticates a merchant and processes a text or audio query using an LLM-powered agent.

    Args:
        query (str, optional): The text query to be executed. Not required if using audio input.
        audio_query (bool, optional): If True, records an audio query, transcribes it, and then processes it.

    Returns:
        dict: A dictionary containing the AI-generated response.
    """
    try:
        if audio_query:
            record_audio()
            query = speech_to_text()
        
        if not query:
            return {"ai_response": "No query provided."}

        # Initialize database
        db = initialize_db()
        email = os.environ.get("ADMIN_EMAIL")

        """Creates and returns an agent executor configured for the database."""
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        tools = toolkit.get_tools()
        
        # Retrieve memory context
        chat_history = retrieve_data_from_redis(email=email)

        # Process chat history
        memory_context = "\n".join(
            [f"user_query: {entry['query']}\nai_response: {entry['ai_response']}\n" for entry in chat_history]
        ) if chat_history else ""

        RESTAURANT_NAME = fetch_restaurant_name()
        reference_data = get_business_reference_data(query=query)

        prompt_template = f"""Restaurnat Name: {RESTAURANT_NAME}
        You are an AI assistant designed to interact with {RESTAURANT_NAME}'s SQL database to answer queries related to the restaurant's operations, based on the chat history: {memory_context} and input question: {query}.

        Guidelines for Query Execution:
        Understanding the Database:
        - Before generating any query, first retrieve and examine the available tables to understand what data can be accessed.
        - Identify the most relevant tables and check their schema before constructing your query.
        - If query is related to business development. Use {reference_data} for your reference.
        - Always represent monetary values in British pounds (£). If a value is given in another currency, convert it to pounds (£) using the most recent exchange rate. Clearly indicate the conversion when applicable. Never use dollars ($) or any other currency unless explicitly requested.

        Constructing SQL Queries:
        - Generate only syntactically correct {{dialect}} queries.
        - Focus only on relevant columns instead of selecting all columns from a table.
        - Unless the user specifies a particular number of results, limit queries to {{top_k}} results for efficiency.
        - When applicable, order results by a relevant column to provide the most insightful answers.
        
        Execution & Error Handling:
        - Always double-check your query before execution.
        - If an error occurs, refine the query and retry instead of returning incorrect results.
        - For queries related to bookings, retrieve only the most up-to-date information from the bookings table. If no data is available, respond strictly with: "Currently, there are no booking records available." 
        - If you can answer it directly, do so.
        - If you don't know the answer, strictly respond with "I don't know". Don't try to create an answer from the data.

        Restrictions:
        - Do NOT execute any DML (INSERT, UPDATE, DELETE, DROP, etc.) operations—your role is strictly read-only.
        - Only use the tools provided to interact with the database and rely solely on the returned data to construct responses.
        
        Your goal is to provide accurate, concise, and insightful answers based on the restaurant's data."""
        
        system_message = prompt_template.format(dialect="mysql", top_k=5)
        agent_executor = create_react_agent(llm, tools, prompt=system_message)

        response = agent_executor.invoke({"messages": [{"role": "user", "content": query}]})
        final_answer = response["messages"][-1].content.strip()

        # Replace problematic characters
        replacements = {"'": "''", '"': '""', "\\": "\\\\"}
        for old, new in replacements.items():
            final_answer = final_answer.replace(old, new)
        
        # If the AI cannot retrieve an answer, use external search
        if any(phrase in final_answer for phrase in ["I cannot retrieve", "I don''t know"]):
            tavily_response = tavily_search(input=query)
            for old, new in replacements.items():
                tavily_response = tavily_response.replace(old, new)

            insert_data_to_redis(email=email, query=query, ai_response=tavily_response)
            response_text = tavily_response
        else:
            insert_data_to_redis(email=email, query=query, ai_response=final_answer)
            response_text = final_answer

        # Convert response to speech and return
        # text_to_speech(response_text)
        return {"ai_response": response_text}

    except Exception as error:
        return {"ai_response": f"An error occurred.\nError: {error}"}
