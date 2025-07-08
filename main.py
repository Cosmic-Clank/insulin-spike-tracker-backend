from services import ai_meal_extract_gpt
from models import AiMealExtractRequest, Meal, ResponseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
import uuid
import time

from utils import save_base64_images


# Load environment variables from .env file

app = FastAPI()

# Allow all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


@app.post("/ai-meal-extract", response_model=ResponseModel)
async def extract_meal(data: AiMealExtractRequest):
    try:
        meal = ai_meal_extract_gpt(data.images, data.textualData)
        meal_id = str(uuid.uuid4())

        save_base64_images(data.images, folder="images")

        return ResponseModel(
            success=True,
            message="Meal extracted successfully",
            data={
                "meal": {
                    "id": meal_id,
                    "name": meal.name,
                    "timestamp": int(time.time() * 1000),
                    "items": [item.model_dump() for item in meal.items],
                }
            }
        )

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
