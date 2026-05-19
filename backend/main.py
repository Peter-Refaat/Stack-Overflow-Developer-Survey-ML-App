from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import os
from predictor import Milestone1Predictor, Milestone2Predictor

app = FastAPI()

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize predictors
# Assuming the pipeline directories are relative to the root or specified here
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
m1_predictor = Milestone1Predictor(os.path.join(BASE_DIR, "saved_pipeline_M1"))
m2_predictor = Milestone2Predictor(os.path.join(BASE_DIR, "saved_pipeline_M2"))

@app.post("/predict")
async def predict(milestone: str = Form(...), file: UploadFile = File(...)):
    if milestone not in ["1", "2"]:
        raise HTTPException(status_code=400, detail="Invalid milestone. Choose 1 or 2.")
    
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    if milestone == "1":
        results = m1_predictor.predict(df)
    else:
        results = m2_predictor.predict(df)

    return {"milestone": milestone, "results": results}

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
