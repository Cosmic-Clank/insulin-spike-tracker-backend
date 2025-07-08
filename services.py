from typing import List

from fastapi import HTTPException
from dotenv import load_dotenv

from models import Meal
from openai import OpenAI
load_dotenv()

client = OpenAI()


def ai_meal_extract_gpt(images: List[str], textual_data: str = "") -> Meal:
    try:
        prompt = """
        You are a nutritionist assistant. Analyze the image of the food and extract the overall meal name and a list of food ingredients that appear in the entire meal.

        For each food ingredient, return the following information as a structured JSON object:

        - **name**: the name of the ingredient (string)
        - **fii**: estimated Food Insulin Index (integer from 0 to 100)
        - **unit**: unit of measurement (one of: "g", "ml", "pcs", "slice", "cup", "tbsp", "serving")
        - **kcalPerUnit**: estimated number of kilocalories for **one unit only**, **not** the total for the whole portion (e.g., 1 slice of bread = 80 kcal, not 160 kcal for 2 slices)
        - **quantity**: how many units were present in the meal (e.g., 2.5 slices, 100g, 1.25 cups)
        - **carb_g**: estimated grams of carbohydrates in the ingredient (integer)
        - **gi**: estimated glycemic index (integer from 0 to 100)
        - **satFat_g**: estimated grams of saturated fat in the ingredient (integer)

        ⚠️ Important:
        - Do **not multiply** kcalPerUnit by quantity — only return kcalPerUnit as the value for **a single unit**.
        - The `quantity` field tells how many units were consumed.
        - We will calculate total kcal programmatically using: `totalKcal = quantity × kcalPerUnit`
        - If images of a nutritional label are provided, use that information as a single item instead of creating multiple items for each ingredient.
        - Make sure to understand the nutitional label correctly. Note the serving size mentioned on the label, note the percentage of the nutional value, and calculate the values accordingly based on how much of the food was consumed.

        Return only the meal name and the array of items in valid JSON format, no text explanation.

        """

        content = [{"type": "input_text", "text": prompt}]

        content += [{"type": "input_image", "image_url": image}
                    for image in images]

        if textual_data:
            content.append({"type": "input_text", "text": textual_data})

        response = client.responses.parse(
            model="gpt-4.1",
            input=[{
                "role": "user",
                "content": content,
            }],  # type: ignore
            text_format=Meal
        )

        parsed_meal = response.output_parsed

        if not parsed_meal:
            raise HTTPException(
                status_code=400, detail="Invalid meal data received")

        return parsed_meal

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")
