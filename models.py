from enum import Enum
from typing import Any, List, Optional

from pydantic import BaseModel


class AiMealExtractRequest(BaseModel):
    images: List[str]
    textualData: str  # Optional, can be used for additional context


class ResponseModel(BaseModel):
    success: bool
    message: str
    data: Optional[Any]


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

    carb_g: int
    gi: int
    satFat_g: int


class Meal(BaseModel):
    id: str  # Same — you can generate in backend
    name: str
    timestamp: int  # You can inject time after parsing
    items: List[MealItem]
