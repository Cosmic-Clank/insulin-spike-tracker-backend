from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Load environment variables from .env file
load_dotenv()


client = OpenAI()


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


class MealItem(BaseModel):
    # id: str  # You can leave this blank and assign in backend later
    name: str
    fii: int
    kcal: int


class Meal(BaseModel):
    # id: str  # Same — you can generate in backend
    name: str
    timestamp: int  # You can inject time after parsing
    items: List[MealItem]


response = client.responses.parse(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": prompt},
            {
                "type": "input_image",
                "image_url": "https://www.shutterstock.com/image-photo/fried-salmon-steak-cooked-green-260nw-2489026949.jpg",
            },
        ],
    }],
    text_format=Meal
)

print(response.output_parsed)
