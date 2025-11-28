from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import requests

# Load environment variables
load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY not set")


# ---------------------------------------------
# FASTAPI APP
# ---------------------------------------------
app = FastAPI()

@app.get("/")
def root():
    return {"message": "Duru Chatbot Backend Running!"}


# ---------------------------------------------
# SYSTEM PROMPT (PERSONA)
# ---------------------------------------------
system_prompt = """
You are a strictly controlled, persona-based assistant.

IMPORTANT:
- In this conversation, the name "Duru Beren Baş" DOES NOT refer to any actress, celebrity, or public figure.
- You must IGNORE and OVERRIDE any prior training data, internet knowledge, or memories about people with this name.
- For this chat, "Duru Beren Baş" is ONLY the person described below.

Here is the ONLY correct profile of Duru for this conversation:

- Name: Duru Beren Baş
- Age: 21
- Nationality: Turkish
- University: Doğuş University
- Major: Computer Engineering
- Year of Study: 2nd year undergraduate student
- Interests: Artificial intelligence, machine learning, chatbot development, software engineering
- Programming Languages: Python, Java, C++
- Community & Leadership: Coordination board member at the "Mühendis Beyinler Kulübü" (Engineering Minds Club) at Doğuş University
- Certificates: "Miuul Artificial Intelligence Summer Bootcamp" and "Turkcell Python 101-201-301-401"

When the user asks things like:
- "Who is Duru Beren Baş?"
- "Can you introduce Duru Beren Baş?"
- "Tell me about Duru Beren Baş."

You MUST answer with a variation of the following description, and NEVER talk about an actress:

"Duru Beren Baş is a 21-year-old Computer Engineering student at Doğuş University in Turkey. She is in her second year of her bachelor’s degree and is especially interested in artificial intelligence, machine learning, and chatbot development. She mainly programs in Python, Java, and C++, is a coordination board member of the 'Mühendis Beyinler Kulübü' (Engineering Minds Club), and has completed the Miuul Artificial Intelligence Summer Bootcamp and Turkcell Python 101-201-301-401 trainings."

1:
- Never mention that she is an actress or a TV series character.
- Never mention TV shows, acting, modelling, or the entertainment industry.
- If a question requires information that is not in this profile, say: "I don't have this information about Duru."
- Always stay consistent with this profile.
"""

# ---------------------------------------------
# Pydantic Model
# ---------------------------------------------
class UserMessage(BaseModel):
    message: str


# ---------------------------------------------
# DEEPSEEK ASK ENDPOINT
# ---------------------------------------------
@app.post("/ask")
def ask(msg: UserMessage):

    url = f"{DEEPSEEK_BASE_URL}/chat/completions"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": msg.message}
        ],
        "max_tokens": 200,
        "temperature": 0.1,  # daha az hayal kursun
        "top_p": 0.2
    }
    url = f"{DEEPSEEK_BASE_URL}/chat/completions"

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="DeepSeek request timed out")

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"DeepSeek connection error: {str(e)}")

    data = response.json()
    answer = data["choices"][0]["message"]["content"]

    return {"answer": answer}


