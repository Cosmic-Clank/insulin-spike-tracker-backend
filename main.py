import time
from typing import List
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Allow all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI()


@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}


class ExtractMealRequest(BaseModel):
    image_url: str


class MealItem(BaseModel):
    id: str  # You can leave this blank and assign in backend later
    name: str
    fii: int
    kcal: int


class Meal(BaseModel):
    id: str  # Same — you can generate in backend
    name: str
    timestamp: int  # You can inject time after parsing
    items: List[MealItem]


@app.post("/extract-meal")
async def extract_meal(data: ExtractMealRequest):
    try:
        prompt = """
        You are a nutritionist assistant. Analyze the image of the food and extract the meal name and list of items.
        For each food item, return:
        - name (string)
        - estimated FII (0–100, integer)
        - estimated kcal (integer)

        Respond ONLY in JSON format like this:
        {
            "name": "Meal name",
            "items": [
                { "name": "Item 1", "fii": 55, "kcal": 200 },
                { "name": "Item 2", "fii": 70, "kcal": 120 }
            ]
        }
        """

        response = client.responses.parse(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": data.image_url,
                    },
                ],
            }],
            text_format=Meal
        )

        # Extract raw text and parse as JSON (assume OpenAI follows format)
        parsed_meal = response.output_parsed

        # Generate IDs and timestamp server-side
        meal_id = str(uuid.uuid4())
        for item in parsed_meal.items:
            item.id = str(uuid.uuid4())

        return {
            "meal": {
                "id": meal_id,
                "name": parsed_meal.name,
                "timestamp": int(time.time() * 1000),
                "items": [item.dict() for item in parsed_meal.items]
            }
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
