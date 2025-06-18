import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

llm = ChatGroq(
    model="llama3-70b-8192",   # going back to the first llm we went with
    temperature=0.7,
    max_tokens=512,
    groq_api_key=os.getenv("GROQ_API_KEY")
)
