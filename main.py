from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import os
from franc import franc

app = FastAPI(title="Lexora-API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Token now read from environment variable – will be set on Render
HF_TOKEN = os.environ["HF_TOKEN"]
HF_API = "https://api-inference.huggingface.co/models"

SUM_MODEL = "facebook/bart-large-cnn"
TRANS_MODELS = {
    "fra": "Helsinki-NLP/opus-mt-fr-en",
    "spa": "Helsinki-NLP/opus-mt-es-en",
    "deu": "Helsinki-NLP/opus-mt-de-en",
    "por": "Helsinki-NLP/opus-mt-pt-en",
    "ita": "Helsinki-NLP/opus-mt-it-en"
}

class DocRequest(BaseModel):
    text: str

async def hf_request(model_id: str, payload: dict):
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{HF_API}/{model_id}",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json=payload,
            timeout=30.0
        )
        if resp.status_code != 200:
            raise HTTPException(500, f"Model error: {resp.text}")
        return resp.json()

@app.post("/summarize")
async def summarise_doc(req: DocRequest):
    # 1. Detect language
    lang = franc(req.text, min_length=3)
    if lang == "und" or lang not in ["eng", "fra", "spa", "deu", "por", "ita"]:
        raise HTTPException(400, f"Unsupported or undetected language: {lang}")

    working_text = req.text
    translated = False

    # 2. Translate if not English
    if lang != "eng":
        result = await hf_request(TRANS_MODELS[lang], {"inputs": working_text})
        working_text = result[0]["translation_text"]
        translated = True

    # 3. Summarize
    summary_result = await hf_request(
        SUM_MODEL,
        {
            "inputs": working_text,
            "parameters": {"max_length": 200, "min_length": 50}
        }
    )
    raw_summary = summary_result[0]["summary_text"]

    # 4. Build bullet points
    bullets = []
    for s in raw_summary.split(". "):
        s = s.strip()
        if len(s) > 10:
            if not s.endswith("."):
                s += "."
            bullets.append(f"- {s}")

    return {
        "detected_language": lang,
        "translated": translated,
        "executive_summary": raw_summary,
        "bullet_points": bullets
    }