from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.model import predict_image
from app.utils import preprocess_image

app = FastAPI(title="COVID-19 X-ray Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to COVID-19 X-ray classifier API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        if not image_bytes:
            return JSONResponse({"error": "Uploaded file is empty."}, status_code=400)

        img_array = preprocess_image(image_bytes)
        label, confidence = predict_image(img_array)

        return JSONResponse({"class": label, "confidence": round(confidence, 4)})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
