"""food_database.py

Centralised nutrition and pricing database for the diet planner.
Keeping this data in its own module lets the main optimisation code stay
clean and makes it easier to update the food information independently.
"""

FOOD_DATABASE = {
    "rice_white": {
        "calories": 130,
        "protein": 2.7,
        "fats": 0.3,
        "carbs": 28.0,
        "vitamins": 0.0,
        "iron": 1.2,
        "cholesterol": 0,
        "cost": 2.99,
    },
    "rice_brown": {
        "calories": 112,
        "protein": 2.32,
        "fats": 0.9,
        "carbs": 23.0,
        "vitamins": 0.0,
        "iron": 0.8,
        "cholesterol": 0,
        "cost": 3.50,
    },
    "bulgur_wheat": {
        "calories": 83,
        "protein": 3.1,
        "fats": 0.2,
        "carbs": 18.6,
        "vitamins": 0.0,
        "iron": 0.9,
        "cholesterol": 0,
        "cost": 2.80,
    },
    "wheat_flour": {
        "calories": 364,
        "protein": 10.3,
        "fats": 1.8,
        "carbs": 76.2,
        "vitamins": 0.0,
        "iron": 3.5,
        "cholesterol": 0,
        "cost": 0.75,
    },
    "bread_baladi": {
        "calories": 265,
        "protein": 9.0,
        "fats": 3.0,
        "carbs": 49.0,
        "vitamins": 0.0,
        "iron": 2.5,
        "cholesterol": 0,
        "cost": 1.50,
    },
    "bread_sliced": {
        "calories": 266,
        "protein": 7.6,
        "fats": 3.3,
        "carbs": 50.6,
        "vitamins": 0.0,
        "iron": 3.7,
        "cholesterol": 0,
        "cost": 6.50,
    },
    "bread_wholewheat": {
        "calories": 247,
        "protein": 13.0,
        "fats": 4.2,
        "carbs": 41.4,
        "vitamins": 0.0,
        "iron": 2.3,
        "cholesterol": 0,
        "cost": 7.00,
    },
    "pasta_macaroni": {
        "calories": 131,
        "protein": 5.0,
        "fats": 1.1,
        "carbs": 25.0,
        "vitamins": 0.0,
        "iron": 0.8,
        "cholesterol": 0,
        "cost": 3.20,
    },
    "oats": {
        "calories": 389,
        "protein": 16.9,
        "fats": 6.9,
        "carbs": 66.3,
        "vitamins": 0.0,
        "iron": 4.7,
        "cholesterol": 0,
        "cost": 2.20,
    },
    "barley_pearl": {
        "calories": 354,
        "protein": 12.5,
        "fats": 2.3,
        "carbs": 73.5,
        "vitamins": 0.0,
        "iron": 3.6,
        "cholesterol": 0,
        "cost": 2.00,
    },
    "lentils": {
        "calories": 352,
        "protein": 24.5,
        "fats": 1.0,
        "carbs": 63.0,
        "vitamins": 0.0,
        "iron": 6.5,
        "cholesterol": 0,
        "cost": 4.00,
    },
    "chickpeas": {
        "calories": 378,
        "protein": 20.0,
        "fats": 6.0,
        "carbs": 63.0,
        "vitamins": 4.0,
        "iron": 4.0,
        "cholesterol": 0,
        "cost": 3.00,
    },
    "fava_beans": {
        "calories": 106,
        "protein": 7.6,
        "fats": 0.4,
        "carbs": 18.0,
        "vitamins": 0.3,
        "iron": 1.5,
        "cholesterol": 0,
        "cost": 3.70,
    },
    "kidney_beans": {
        "calories": 127,
        "protein": 8.7,
        "fats": 0.5,
        "carbs": 22.8,
        "vitamins": 0.0,
        "iron": 2.5,
        "cholesterol": 0,
        "cost": 3.00,
    },
    "peas_green": {
        "calories": 81,
        "protein": 5.4,
        "fats": 0.4,
        "carbs": 14.5,
        "vitamins": 40.0,
        "iron": 1.2,
        "cholesterol": 0,
        "cost": 3.00,
    },
    "potato": {
        "calories": 79,
        "protein": 2.0,
        "fats": 0.1,
        "carbs": 17.0,
        "vitamins": 16.0,
        "iron": 0.8,
        "cholesterol": 0,
        "cost": 4.00,
    },
    "onion": {
        "calories": 40,
        "protein": 1.1,
        "fats": 0.1,
        "carbs": 9.3,
        "vitamins": 8.0,
        "iron": 0.2,
        "cholesterol": 0,
        "cost": 1.00,
    },
    "tomato": {
        "calories": 18,
        "protein": 0.9,
        "fats": 0.2,
        "carbs": 3.9,
        "vitamins": 13.8,
        "iron": 0.3,
        "cholesterol": 0,
        "cost": 3.00,
    },
    "cucumber": {
        "calories": 16,
        "protein": 0.7,
        "fats": 0.1,
        "carbs": 3.6,
        "vitamins": 2.8,
        "iron": 0.3,
        "cholesterol": 0,
        "cost": 1.50,
    },
    "carrot": {
        "calories": 41,
        "protein": 0.9,
        "fats": 0.2,
        "carbs": 9.6,
        "vitamins": 6.7,
        "iron": 0.3,
        "cholesterol": 0,
        "cost": 2.00,
    },
    "eggplant": {
        "calories": 25,
        "protein": 1.0,
        "fats": 0.2,
        "carbs": 5.9,
        "vitamins": 3.0,
        "iron": 0.2,
        "cholesterol": 0,
        "cost": 3.00,
    },
    "okra": {
        "calories": 33,
        "protein": 2.0,
        "fats": 0.2,
        "carbs": 7.5,
        "vitamins": 23.0,
        "iron": 0.6,
        "cholesterol": 0,
        "cost": 5.00,
    },
    "pepper_green": {
        "calories": 20,
        "protein": 0.9,
        "fats": 0.2,
        "carbs": 4.6,
        "vitamins": 80.0,
        "iron": 0.3,
        "cholesterol": 0,
        "cost": 5.00,
    },
    "pepper_red": {
        "calories": 31,
        "protein": 1.0,
        "fats": 0.3,
        "carbs": 6.0,
        "vitamins": 65.0,
        "iron": 0.4,
        "cholesterol": 0,
        "cost": 6.00,
    },
    "spinach": {
        "calories": 23,
        "protein": 2.9,
        "fats": 0.4,
        "carbs": 3.6,
        "vitamins": 28.5,
        "iron": 2.7,
        "cholesterol": 0,
        "cost": 2.50,
    },
    "cabbage": {
        "calories": 25,
        "protein": 1.3,
        "fats": 0.1,
        "carbs": 5.8,
        "vitamins": 37.0,
        "iron": 0.5,
        "cholesterol": 0,
        "cost": 1.50,
    },
    "cauliflower": {
        "calories": 25,
        "protein": 1.9,
        "fats": 0.3,
        "carbs": 5.0,
        "vitamins": 48.0,
        "iron": 0.4,
        "cholesterol": 0,
        "cost": 4.00,
    },
    "lettuce": {
        "calories": 15,
        "protein": 1.4,
        "fats": 0.1,
        "carbs": 2.9,
        "vitamins": 9.0,
        "iron": 0.3,
        "cholesterol": 0,
        "cost": 2.00,
    },
    "zucchini": {
        "calories": 17,
        "protein": 1.2,
        "fats": 0.3,
        "carbs": 3.1,
        "vitamins": 17.0,
        "iron": 0.4,
        "cholesterol": 0,
        "cost": 2.50,
    },
    "banana": {
        "calories": 89,
        "protein": 1.1,
        "fats": 0.3,
        "carbs": 22.8,
        "vitamins": 8.7,
        "iron": 0.3,
        "cholesterol": 0,
        "cost": 4.00,
    },
    "orange": {
        "calories": 47,
        "protein": 0.9,
        "fats": 0.1,
        "carbs": 11.8,
        "vitamins": 53.2,
        "iron": 0.1,
        "cholesterol": 0,
        "cost": 5.00,
    },
    "apple": {
        "calories": 52,
        "protein": 0.3,
        "fats": 0.2,
        "carbs": 13.8,
        "vitamins": 4.6,
        "iron": 0.1,
        "cholesterol": 0,
        "cost": 7.00,
    },
    "grapes": {
        "calories": 69,
        "protein": 0.7,
        "fats": 0.2,
        "carbs": 18.1,
        "vitamins": 3.2,
        "iron": 0.4,
        "cholesterol": 0,
        "cost": 10.00,
    },
    "mango": {
        "calories": 60,
        "protein": 0.8,
        "fats": 0.4,
        "carbs": 15.0,
        "vitamins": 36.4,
        "iron": 0.1,
        "cholesterol": 0,
        "cost": 4.00,
    },
    "dates": {
        "calories": 282,
        "protein": 2.5,
        "fats": 0.4,
        "carbs": 75.0,
        "vitamins": 0.4,
        "iron": 0.9,
        "cholesterol": 0,
        "cost": 5.00,
    },
    "guava": {
        "calories": 68,
        "protein": 2.6,
        "fats": 1.0,
        "carbs": 14.3,
        "vitamins": 228.0,
        "iron": 0.3,
        "cholesterol": 0,
        "cost": 6.00,
    },
    "figs": {
        "calories": 74,
        "protein": 0.8,
        "fats": 0.3,
        "carbs": 19.2,
        "vitamins": 2.0,
        "iron": 0.4,
        "cholesterol": 0,
        "cost": 8.00,
    },
    "peanut_raw": {
        "calories": 567,
        "protein": 25.8,
        "fats": 49.2,
        "carbs": 16.1,
        "vitamins": 0.0,
        "iron": 4.6,
        "cholesterol": 0,
        "cost": 12.00,
    },
    "almond_raw": {
        "calories": 579,
        "protein": 21.2,
        "fats": 49.9,
        "carbs": 21.6,
        "vitamins": 0.0,
        "iron": 3.7,
        "cholesterol": 0,
        "cost": 20.00,
    },
    "pistachio_raw": {
        "calories": 562,
        "protein": 20.3,
        "fats": 45.3,
        "carbs": 27.5,
        "vitamins": 0.0,
        "iron": 4.0,
        "cholesterol": 0,
        "cost": 30.00,
    },
    "sunflower_oil": {
        "calories": 884,
        "protein": 0.0,
        "fats": 100.0,
        "carbs": 0.0,
        "vitamins": 0.0,
        "iron": 0.0,
        "cholesterol": 0,
        "cost": 9.00,
    },
    "olive_oil": {
        "calories": 900,
        "protein": 0.0,
        "fats": 100.0,
        "carbs": 0.0,
        "vitamins": 0.0,
        "iron": 0.0,
        "cholesterol": 0,
        "cost": 31.30,
    },
    "sugar": {
        "calories": 387,
        "protein": 0.0,
        "fats": 0.0,
        "carbs": 100.0,
        "vitamins": 0.0,
        "iron": 0.0,
        "cholesterol": 0,
        "cost": 2.50,
    },
    "honey": {
        "calories": 304,
        "protein": 0.3,
        "fats": 0.0,
        "carbs": 82.4,
        "vitamins": 0.0,
        "iron": 0.4,
        "cholesterol": 0,
        "cost": 25.00,
    },
    "whole_milk": {
        "calories": 60,
        "protein": 3.3,
        "fats": 3.3,
        "carbs": 5.0,
        "vitamins": 0.0,
        "iron": 0.0,
        "cholesterol": 10,
        "cost": 2.10,
    },
    "yogurt_plain": {
        "calories": 61,
        "protein": 3.5,
        "fats": 3.3,
        "carbs": 4.7,
        "vitamins": 0.0,
        "iron": 0.0,
        "cholesterol": 5,
        "cost": 1.50,
    },
    "white_cheese": {
        "calories": 270,
        "protein": 18.0,
        "fats": 22.0,
        "carbs": 2.0,
        "vitamins": 0.5,
        "iron": 0.5,
        "cholesterol": 60,
        "cost": 20.00,
    },
    "feta_cheese": {
        "calories": 264,
        "protein": 14.2,
        "fats": 21.0,
        "carbs": 4.1,
        "vitamins": 0.5,
        "iron": 0.1,
        "cholesterol": 89,
        "cost": 25.00,
    },
    "butter": {
        "calories": 717,
        "protein": 0.85,
        "fats": 81.1,
        "carbs": 0.06,
        "vitamins": 0.7,
        "iron": 0.02,
        "cholesterol": 215,
        "cost": 25.00,
    },
    "egg_chicken": {
        "calories": 143,
        "protein": 12.6,
        "fats": 9.5,
        "carbs": 0.7,
        "vitamins": 0.6,
        "iron": 1.8,
        "cholesterol": 372,
        "cost": 10.00,
    },
    "chicken_breast": {
        "calories": 120,
        "protein": 22.5,
        "fats": 2.6,
        "carbs": 0.0,
        "vitamins": 0.0,
        "iron": 0.37,
        "cholesterol": 73,
        "cost": 12.00,
    },
    "chicken_thigh": {
        "calories": 119,
        "protein": 18.0,
        "fats": 5.0,
        "carbs": 0.0,
        "vitamins": 0.0,
        "iron": 0.8,
        "cholesterol": 90,
        "cost": 10.00,
    },
    "beef_lean": {
        "calories": 176,
        "protein": 17.0,
        "fats": 10.0,
        "carbs": 0.0,
        "vitamins": 0.0,
        "iron": 1.9,
        "cholesterol": 55,
        "cost": 45.00,
    },
    "beef_fatty": {
        "calories": 250,
        "protein": 20.0,
        "fats": 20.0,
        "carbs": 0.0,
        "vitamins": 0.0,
        "iron": 2.6,
        "cholesterol": 70,
        "cost": 50.00,
    },
    "lamb": {
        "calories": 294,
        "protein": 25.0,
        "fats": 21.0,
        "carbs": 0.0,
        "vitamins": 0.0,
        "iron": 2.1,
        "cholesterol": 76,
        "cost": 50.00,
    },
    "fish_tilapia": {
        "calories": 96,
        "protein": 20.0,
        "fats": 1.7,
        "carbs": 0.0,
        "vitamins": 0.0,
        "iron": 0.6,
        "cholesterol": 50,
        "cost": 10.00,
    },
    "fish_sardine": {
        "calories": 208,
        "protein": 24.6,
        "fats": 11.5,
        "carbs": 0.0,
        "vitamins": 0.0,
        "iron": 2.9,
        "cholesterol": 142,
        "cost": 8.00,
    },
}

# List of food items for easy indexing
FOOD_ITEMS = list(FOOD_DATABASE.keys())
