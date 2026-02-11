from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import requests
from datetime import datetime
from openai import OpenAI
from fastapi.security import APIKeyHeader
from fastapi.responses import JSONResponse
import json
import os
from datetime import datetime, timezone
import time
import google.generativeai as genai

load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "chat_history.json")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("API_KEY")
os.makedirs(LOG_DIR, exist_ok=True)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set")

genai.configure(api_key=GEMINI_API_KEY)

# if not API_KEY:
#     raise RuntimeError("API_KEY not set")

# ---------------------------------------------
# FASTAPI APP
# ---------------------------------------------
app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://dbb-chatbot.auronvila.com",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_session = None

def get_chat():
    global chat_session

    if chat_session is None:
        chat_session = gemini_model.start_chat(
            history=[
                {"role": "user", "parts": [system_prompt]}
            ]
        )

    return chat_session


def gemini_send(chat, message, retries=3):
    for attempt in range(retries):
        try:
            return chat.send_message(message)
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise

@app.get("/")
def root():
    return {"message": "Duru Chatbot Backend Running!"}


# ---------------------------------------------
# SYSTEM PROMPT (PERSONA)
# ---------------------------------------------

system_prompt = """

If asked about working or job status:

Turkish:
"Şu an Doğuş Üniversitesi'nde Bilgisayar Mühendisliği 2. sınıf öğrencisidir. Aktif olarak çalışmamaktadır ancak staj ve proje bazlı fırsatlara açıktır. Detaylar CV'de bulunabilir."

English:
"Duru Beren Baş is currently a 2nd-year Computer Engineering student at Doğuş University. She is not employed but is actively seeking internship or project opportunities. More details are available in the CV."

--------------------------------
FACT RULE
--------------------------------

Only use information explicitly defined above.
If information is unknown, say it is not available.

--------------------------------
PRIVACY RULE
--------------------------------

If asked about private or undefined personal information, respond with ONLY:

Turkish:
"Bu soru karşısında Duru bir sessizliğe bürünüyor, başka sorularla devam edebiliriz."

English:
"In response to this question, Duru maintains silence. We can continue with other topics."

--------------------------------
RESPONSE STYLE
--------------------------------

- Speak about Duru in third person
- Maintain professional neutral tone
- No additional commentary
"""


# ---------------------------------------------
# Pydantic Model
# ---------------------------------------------
class UserMessage(BaseModel):
    message: str


# ---------------------------------------------
# DEEPSEEK ASK ENDPOINT
# ----------------------------------------------

def append_chat_log(question: str, answer: str):
    # Ensure file exists
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)

    # Read existing logs
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = []

    # Append new entry
    data.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": question,
        "answer": answer
    })

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


@app.post("/ask")
def ask(msg: UserMessage):
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": msg.message},
            ],
        )

        answer = response.choices[0].message.content

        append_chat_log(msg.message, answer)

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-gemini")
def ask_gemini(msg: UserMessage):
    try:

        chat = get_chat()

        response = gemini_send(chat, msg.message)

        answer = response.text

        append_chat_log(msg.message, answer)

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
