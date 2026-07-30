import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# Match the exact names from your .env file
gemini_key = os.getenv("GEMINI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")
gpt_key = os.getenv("GPT_API_KEY")
claude_key = os.getenv("CLAUDE_API_KEY")
mistral_key = os.getenv("MISTRAL_API_KEY")

if gemini_key and groq_key and gpt_key and claude_key and mistral_key:
    print("✅ Success: .env file found and ALL 5 keys are loaded perfectly!")
else:
    print("❌ Error: One or more keys missing. Double check variable names.")