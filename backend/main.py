from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from backend.model_logic import predict_publishability
from backend.conference_recommender import recommend_conference
import fitz
import re

app = FastAPI()

class PaperInput(BaseModel):
    text: str

@app.post("/predict_publishability")
def predict(input_data: PaperInput):
    result, confidence = predict_publishability(input_data.text)
    return JSONResponse(content={"result": result, "confidence": confidence})

@app.post("/predict_conference")
def predict_full(input_data: PaperInput):
    result, _ = predict_publishability(input_data.text)
    if result == "publishable":
        conf, rationale = recommend_conference(input_data.text)
        return JSONResponse(content={"publishable": True, "conference": conf, "rationale": rationale})
    else:
        return JSONResponse(content={"publishable": False, "conference": None, "rationale": "Paper is not publishable."})

def clean_text(text):
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'References.*', '', text, flags=re.I)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    with open("temp.pdf", "wb") as f:
        f.write(contents)

    text = ""
    with fitz.open("temp.pdf") as doc:
        for page in doc:
            text += page.get_text()

    text = clean_text(text)

    result, _ = predict_publishability(text)
    if result == "publishable":
        conf, rationale = recommend_conference(text)
        return JSONResponse(content={"publishable": True, "conference": conf, "rationale": rationale})
    else:
        return JSONResponse(content={"publishable": False, "conference": None, "rationale": "Paper is not publishable."})