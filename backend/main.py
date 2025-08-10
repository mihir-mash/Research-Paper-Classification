from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from model2.predict_and_explain import full_pipeline 
import fitz 
import re
from fastapi.middleware.cors import CORSMiddleware
import os 

app = FastAPI()

origins = [
    "http://127.0.0.1:5500",  
    "http://localhost:5500",   
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],    
    allow_headers=["*"],   
)

class PaperInput(BaseModel):
    text: str

def clean_text(text: str) -> str:

    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'References.*', '', text, flags=re.I | re.DOTALL) 
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) # Remove non-ASCII characters
    return text.strip()

@app.post("/predict_conference_full")
def predict_conference_full(input_data: PaperInput):

    try:
        result = full_pipeline(input_data.text)
        return JSONResponse(content=result)
    except Exception as e:
        # This will catch errors from the model pipeline
        raise HTTPException(status_code=500, detail=f"Model pipeline error: {str(e)}")

# Predict from uploaded PDF 
@app.post("/upload_pdf_full")
async def upload_pdf_full(file: UploadFile = File(...)):

    try:

        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

        contents = await file.read()

        text = ""
        with fitz.open(stream=contents, filetype="pdf") as doc:

            if doc.page_count == 0:
                raise HTTPException(status_code=400, detail="Could not extract any pages from the PDF. It may be empty or corrupted.")
            
            for page in doc:
                text += page.get_text()

        if not text.strip():
             raise HTTPException(status_code=400, detail="The PDF appears to contain no extractable text (e.g., it might be image-only).")

        cleaned_text = clean_text(text)

        result = full_pipeline(cleaned_text)

        return JSONResponse(content=result)

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")