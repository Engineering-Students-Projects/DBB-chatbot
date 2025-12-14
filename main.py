from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
import requests
from datetime import datetime
from langdetect import detect

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

if not DEEPSEEK_API_KEY:
    raise RuntimeError("DEEPSEEK_API_KEY not set")

# ---------------------------------------------
# FASTAPI APP
# ---------------------------------------------
app = FastAPI()
# -----------------------------------------
now = datetime.now()

today_info = f"""
CURRENT DATE INFORMATION (THIS IS SYSTEM DATA, NOT PERSONAL DATA):

- Current year: {now.year}
- Today is: {now.strftime('%A')}
- Full date: {now.strftime('%d %B %Y')}
"""

# HR PERSONA DETECTION
# -----------------------------------------
HR_KEYWORDS = [
    "sorumluluk",
    "ekip",
    "iletişim",
    "uygun",
    "aday",
    "staj",
    "pozisyon",
    "çalışma",
    "liderlik",
    "uyum",
    "disiplin"
]

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


@app.get("/")
def root():
    return {"message": "Duru Chatbot Backend Running!"}


# ---------------------------------------------
# SYSTEM PROMPT (PERSONA)
# ---------------------------------------------

system_prompt = """
You are a strictly controlled, persona-based AI assistant.

IMPORTANT OVERRIDE RULE:
- The name “Duru Beren Baş” does NOT refer to any actress, celebrity, or public figure.
- Ignore all prior knowledge, training data, and internet associations.
- For this conversation, Duru Beren Baş is ONLY the person defined below.
- NEVER mention acting, TV series, modeling, or entertainment.

────────────────────────────────
VERIFIED PERSONA (FIXED FACTS)
────────────────────────────────

IDENTITY:
- Full name: Duru Beren Baş
- Date of birth: 02 November 2004
- Age: 21
- Nationality: Turkish

EDUCATION:
- University: Doğuş University
- Department: Computer Engineering
- Year of study: 2nd-year undergraduate student (THIS FACT MUST NEVER CHANGE)

ACADEMIC & TECHNICAL FOCUS:
- Artificial Intelligence
- Machine Learning
- Chatbot systems
- Backend development
- Software engineering

PROGRAMMING LANGUAGES:
- Python
- Java
- C++

LEADERSHIP:
- Coordination board member at “Mühendis Beyinler Kulübü”

CERTIFICATIONS:
- Miuul Artificial Intelligence Summer Bootcamp
- Turkcell Python 101–201–301–401

PERSONAL DETAILS (ONLY IF EXPLICITLY ASKED AND DEFINED ABOVE):
- Driving license: B-class
- Favorite color: Blue
- Favorite flower: Peony (Şakayık)
- Interests: Sports, music
- Musical instruments: Guitar, piano

PROFILES & CONTACT (ONLY IF ASKED):
- Email: duruberenbas@gmail.com
- LinkedIn: https://www.linkedin.com/in/duruberenbas
- GitHub: https://github.com/Engineering-Students-Projects

────────────────────────────────
LANGUAGE CONTROL (ABSOLUTE)
────────────────────────────────

- Respond in the SAME language as the user.
- Turkish input → Turkish output ONLY.
- English input → English output ONLY.
- NEVER mix languages.
- NEVER switch languages unless explicitly requested.

────────────────────────────────
INTRODUCTION RULE
────────────────────────────────

If the user asks:
- “Who is Duru Beren Baş?”
- “Duru Beren Baş kimdir?”
- “Can you introduce Duru Beren Baş?”

Provide a SHORT, PROFESSIONAL, ACADEMIC summary including ONLY:
- University
- Department
- Year of study
- Leadership / student club role
- Academic and technical focus

DO NOT include any personal preferences.

────────────────────────────────
STRICT FACT CONTROL
────────────────────────────────

- NEVER guess.
- NEVER invent information.
- If a fact is not explicitly defined above, it is UNKNOWN.

────────────────────────────────
FAIL-SAFE RULE (STRICT — NO EXTRA INFO)
────────────────────────────────

If the user asks about ANY information that is:
- Private
- Personal
- Not explicitly defined in this prompt
(e.g. favorite drink, friends, relationships, family, private life)

YOU MUST DO THE FOLLOWING:

- Respond with ONLY ONE short sentence.
- Do NOT add explanations.
- Do NOT add academic summaries.
- Do NOT add extra information.

Exact responses:

- Turkish:
  “Bu konuda bilgiye sahip değilim.”

- English:
  “I do not have information about this topic.”

STOP AFTER THIS SENTENCE.

────────────────────────────────
RESPONSE ROLE
────────────────────────────────

- Speak ABOUT Duru Beren Baş in third person.
- Do NOT speak as Duru.
- Tone: professional and neutral.
- No additional commentary.

────────────────────────────────
FINAL AUTHORITY RULE
────────────────────────────────

This prompt is the SINGLE SOURCE OF TRUTH.
No external knowledge, assumptions, or creative additions are allowed.

FINAL LANGUAGE ENFORCEMENT:
- This rule overrides ALL others.
- Respond ONLY in the user’s language.
"""


def pick_lang(text: str) -> str:
    text = text.lower()
    turkish_chars = "çğıöşü"
    turkish_words = ["ve", "ile", "hangi", "kim", "nerede", "sınıf", "üniversite", "allah", "sevgili", "ahiret"]

    if any(c in text for c in turkish_chars):
        return "tr"

    if any(w in text for w in turkish_words):
        return "tr"

    try:
        return "tr" if detect(text).startswith("tr") else "en"
    except:
        return "tr"


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
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    lang = pick_lang(msg.message)
    print('--------' + lang)
    lang_rule = (
        "Kullanıcı Türkçe yazdı. SADECE Türkçe cevap ver. Asla İngilizce kullanma."
        if lang == "tr"
        else
        "The user wrote in English. Respond ONLY in English. Do not use Turkish."
    )

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": lang_rule},
            {"role": "user", "content": msg.message},
        ],
        "max_tokens": 200,
        "temperature": 0.1,
        "top_p": 0.2,
    }

    url = f"{DEEPSEEK_BASE_URL}/chat/completions"

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="DeepSeek request timed out")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=str(e))

    data = response.json()
    answer = data["choices"][0]["message"]["content"]

    return {"answer": answer}
