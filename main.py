import time
from typing import List
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os
from enum import Enum
import base64

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
    images: List[str]
    textualData: str  # Optional, can be used for additional context


class Unit(str, Enum):
    Grams = "g"
    Milliliters = "ml"
    Pieces = "pcs"
    Slices = "slice"
    Cups = "cup"
    Tablespoons = "tbsp"
    Servings = "serving"


class MealItem(BaseModel):
    id: str
    name: str
    fii: int
    quantity: float
    unit: Unit
    kcalPerUnit: int


class Meal(BaseModel):
    id: str  # Same — you can generate in backend
    name: str
    timestamp: int  # You can inject time after parsing
    items: List[MealItem]
    aiComment: str


@app.post("/extract-meal")
async def extract_meal(data: ExtractMealRequest):
    try:
        prompt = """
        You are a nutritionist assistant. Analyze the image of the food and extract the overall meal name and a list of food ingredients that appear in the entire meal.

For each food ingredient, return the following information as a structured JSON object:

- **name**: the name of the ingredient (string)
- **fii**: estimated Food Insulin Index (integer from 0 to 100)
- **unit**: unit of measurement (one of: "g", "ml", "pcs", "slice", "cup", "tbsp", "serving")
- **kcalPerUnit**: estimated number of kilocalories for **one unit only**, **not** the total for the whole portion (e.g., 1 slice of bread = 80 kcal, not 160 kcal for 2 slices)
- **quantity**: how many units were present in the meal (e.g., 2.5 slices, 100g, 1.25 cups)
- **aiComment**: talk about what u understand about the meal, and talk about the healthiness of the meal, and what you would recommend to improve it, and talk about the insuline and glucose response of the meal.

⚠️ Important:
- Do **not multiply** kcalPerUnit by quantity — only return kcalPerUnit as the value for **a single unit**.
- The `quantity` field tells how many units were consumed.
- We will calculate total kcal programmatically using: `totalKcal = quantity × kcalPerUnit`

Return only the meal name and the array of items in valid JSON format, no text explanation.

        """

        content = [{"type": "input_text", "text": prompt}]

        content += [{"type": "input_image", "image_url": image}
                    for image in data.images]

        if data.textualData:
            content.append({"type": "input_text", "text": data.textualData})

        for image in data.images:
            # Ensure the images directory exists
            os.makedirs("images", exist_ok=True)
            # Determine the image index (1-based)
            image_index = data.images.index(image) + 1
            # Decode base64 image and save to file
            image_data = base64.b64decode(image.split(",")[-1])
            with open(f"images/{image_index}.jpg", "wb") as f:
                f.write(image_data)

        response = client.responses.parse(
            model="gpt-4.1",
            input=[{
                "role": "user",
                "content": content,
            }],  # type: ignore
            text_format=Meal
        )

        # Extract raw text and parse as JSON (assume OpenAI follows format)
        parsed_meal = response.output_parsed

        # Generate IDs and timestamp server-side
        if not parsed_meal:
            raise HTTPException(
                status_code=400, detail="Invalid meal data received")

        meal_id = str(uuid.uuid4())
        for item in parsed_meal.items:
            item.id = str(uuid.uuid4())

        return {
            "meal": {
                "id": meal_id,
                "name": parsed_meal.name,
                "timestamp": int(time.time() * 1000),
                "items": [item.model_dump() for item in parsed_meal.items],
                "aiComment": parsed_meal.aiComment
            }
        }

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
