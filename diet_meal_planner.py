import numpy as np
import pandas as pd
from typing import Dict, Any

# Food database with nutrition info and costs


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


def calculate_daily_needs(user_profile: Dict[str, Any]) -> Dict[str, float]:
    """Calculate daily nutritional requirements based on user profile."""
    age = user_profile["age"]
    gender = user_profile["gender"]
    weight = user_profile["weight"]  # kg
    height = user_profile["height"]  # cm
    activity_level = user_profile["activity_level"]
    goal = user_profile["goal"]

    # Calculate BMR (Basal Metabolic Rate) using Mifflin-St Jeor equation
    if gender.lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    # Activity level multipliers
    activity_multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very active": 1.9,
    }

    # Calculate TDEE (Total Daily Energy Expenditure)
    tdee = bmr * activity_multipliers[activity_level.lower()]

    # Adjust calories based on goal
    if goal.lower() == "lose":
        calories = tdee - 500  # 500 calorie deficit for weight loss
    elif goal.lower() == "gain":
        calories = tdee + 500  # 500 calorie surplus for weight gain
    else:  # maintain
        calories = tdee
    print("TARGETED Calories:", calories)
    # Calculate macronutrient needs
    if goal.lower() == "gain":
        protein_ratio = 0.15
        fat_ratio = 0.35
        carb_ratio = 0.5
    elif goal.lower() == "lose":
        protein_ratio = 0.25
        fat_ratio = 0.25
        carb_ratio = 0.5
    else:  # maintain
        protein_ratio = 0.35
        fat_ratio = 0.20
        carb_ratio = 0.45

    protein_calories = calories * protein_ratio
    fat_calories = calories * fat_ratio
    carb_calories = calories * carb_ratio

    protein_grams = protein_calories / 4  # 4 calories per gram of protein
    fat_grams = fat_calories / 9  # 9 calories per gram of fat
    carb_grams = carb_calories / 4  # 4 calories per gram of carbs

    # Other nutrient needs (simplified estimates)
    if gender.lower() == "male":
        iron_mg = 8 if age > 50 else 8
        cholesterol_max = 300  # mg
    else:
        iron_mg = 8 if age > 50 else 18
        cholesterol_max = 300  # mg

    vitamins_arbitrary_units = 5  # Just a placeholder value

    return {
        "calories": calories,
        "protein": protein_grams,
        "fats": fat_grams,
        "carbs": carb_grams,
        "vitamins": vitamins_arbitrary_units,
        "iron": iron_mg,
        "cholesterol": cholesterol_max,
    }


def initialize_population(pop_size, num_foods, min_portion=0, max_portion=300):
    """
    Initialize a population of daily meal plans.
    Each individual is a 1D array of food portions in grams.
    Shape: (pop_size, num_foods)
    """
    population = []
    for _ in range(pop_size):
        # Daily plan is a 1D array
        daily_plan = np.random.uniform(min_portion, max_portion, num_foods)
        population.append(daily_plan)
    return np.array(population)


def calculate_nutrition_and_cost_for_day(daily_chromosome_portions):
    """Calculate total nutrition and cost for a single day's meal plan (1D chromosome)."""
    daily_totals = {
        "calories": 0,
        "protein": 0,
        "fats": 0,
        "carbs": 0,
        "vitamins": 0,
        "iron": 0,
        "cholesterol": 0,
        "cost": 0,
    }

    for i, portion in enumerate(daily_chromosome_portions):
        if portion > 0:
            food_name = FOOD_ITEMS[i]  # FOOD_ITEMS must be globally defined
            food_data = FOOD_DATABASE[
                food_name
            ]  # FOOD_DATABASE must be globally defined

            factor = portion / 100.0
            for (
                nutrient_key
            ) in daily_totals.keys():  # Iterate over keys in daily_totals
                if nutrient_key in food_data:
                    daily_totals[nutrient_key] += food_data[nutrient_key] * factor
    return daily_totals


# Replace the existing calculate_fitness function
def calculate_fitness(
    daily_chromosome,  # Renamed from weekly_chromosome
    requirements: Dict[str, float],
    user_profile: Dict[str, Any],
    generation: int,  # Gen
    max_generations: int,  # Gen_max
):
    """
    Calculate the fitness of a daily meal plan based on the provided mathematical formulation.
    Fitness_GA(x, Gen) = -ActualCost(x) - (PW(Gen) * TotalNutrientBasePenalty(x) + P_small_portions_base(x))
    """
    # 1. Calculate Actual_k(x) and ActualCost(x)
    # daily_chromosome is 'x' in the formulation
    actual_nutrition_and_cost = calculate_nutrition_and_cost_for_day(daily_chromosome)
    actual_cost = actual_nutrition_and_cost.pop(
        "cost"
    )  # Remove cost for nutrient penalty calculation

    # 2. Calculate Base Penalty Functions (Unscaled)
    total_nutrient_base_penalty = 0
    goal = user_profile["goal"].lower()  # GoalUser

    # Define BasePenaltyMult_k_condition and Thresh_k_condition (as per your previous script logic)
    # These are the multipliers and thresholds for penalties BEFORE PW(Gen) scaling.
    # Example for calories (you'll need to define these for all nutrients based on your old penalty structure)
    # This section needs to mirror the penalty logic from your original daily planner's fitness function.

    # --- Calorie Base Penalty P_calories_base(x) ---
    ac_calories = actual_nutrition_and_cost.get("calories", 0)
    tc_calories = requirements.get(
        "calories", 1
    )  # Target_calories, avoid division by zero
    p_calories_base = 0
    # These multipliers (100, 60, 80 etc.) are BasePenaltyMult_k_condition
    if goal == "lose":
        thresh_over = tc_calories * 1.02  # Thresh_calories_lose_over
        thresh_under = tc_calories * 0.95  # Thresh_calories_lose_under
        if ac_calories > thresh_over:
            dev_over = (ac_calories - thresh_over) / tc_calories
            p_calories_base += 100 * dev_over**2  # BasePenaltyMult_cal_lose_over
        elif ac_calories < thresh_under:
            dev_under = (
                thresh_under - ac_calories
            ) / thresh_under  # Denominator as per your previous stricter version
            p_calories_base += 60 * dev_under**2  # BasePenaltyMult_cal_lose_under
    elif goal == "gain":
        thresh_under = tc_calories * 0.98
        thresh_over = tc_calories * 1.10
        if ac_calories < thresh_under:
            dev_under = (thresh_under - ac_calories) / thresh_under
            p_calories_base += 100 * dev_under**2
        elif ac_calories > thresh_over:
            dev_over = (ac_calories - thresh_over) / tc_calories
            p_calories_base += 40 * dev_over**2
    else:  # maintain
        thresh_lower = tc_calories * 0.95
        thresh_upper = tc_calories * 1.05
        if not (thresh_lower <= ac_calories <= thresh_upper):
            dev = abs(ac_calories - tc_calories) / tc_calories
            p_calories_base += 80 * dev**2
    total_nutrient_base_penalty += p_calories_base

    # --- Protein Base Penalty P_protein_base(x) ---
    ac_protein = actual_nutrition_and_cost.get("protein", 0)
    tc_protein = requirements.get("protein", 1)
    p_protein_base = 0
    thresh_prot_under = tc_protein * 0.90
    thresh_prot_over = tc_protein * 1.30
    if ac_protein < thresh_prot_under:
        dev_under = (thresh_prot_under - ac_protein) / thresh_prot_under
        p_protein_base += 20 * dev_under**2
    elif ac_protein > thresh_prot_over:
        dev_over = (ac_protein - thresh_prot_over) / tc_protein
        p_protein_base += 5 * dev_over**2
    total_nutrient_base_penalty += p_protein_base

    # --- Fats Base Penalty P_fats_base(x) ---
    ac_fats = actual_nutrition_and_cost.get("fats", 0)
    tc_fats = requirements.get("fats", 1)
    p_fats_base = 0
    thresh_fats_over = tc_fats * 1.15
    thresh_fats_under = tc_fats * 0.70  # if fats are essential
    if ac_fats > thresh_fats_over:
        dev_over = (ac_fats - thresh_fats_over) / tc_fats
        p_fats_base += 15 * dev_over**2
    elif ac_fats < thresh_fats_under:  # Assuming some fats are needed
        dev_under = (thresh_fats_under - ac_fats) / thresh_fats_under
        p_fats_base += 10 * dev_under**2
    total_nutrient_base_penalty += p_fats_base

    # --- Cholesterol Base Penalty P_cholesterol_base(x) ---
    ac_chol = actual_nutrition_and_cost.get("cholesterol", 0)
    tc_chol = requirements.get("cholesterol", 1)  # Max target
    p_chol_base = 0
    thresh_chol_over = tc_chol * 1.15  # Max is 1.0 * target, so 1.15 is over
    if ac_chol > thresh_chol_over:  # Only penalize if over
        dev_over = (
            ac_chol - thresh_chol_over
        ) / tc_chol  # Denominator could be tc_chol or thresh_chol_over
        p_chol_base += 15 * dev_over**2  # Using same multiplier as fats for "over"
    total_nutrient_base_penalty += p_chol_base

    # --- Vitamins Base Penalty P_vitamins_base(x) ---
    ac_vit = actual_nutrition_and_cost.get("vitamins", 0)
    tc_vit = requirements.get("vitamins", 1)
    p_vit_base = 0
    thresh_vit_under = tc_vit * 0.70
    thresh_vit_over = tc_vit * 1.50  # Wider range for vitamins
    if ac_vit < thresh_vit_under:
        dev_under = (thresh_vit_under - ac_vit) / thresh_vit_under
        p_vit_base += 5 * dev_under**2
    elif ac_vit > thresh_vit_over:
        dev_over = (ac_vit - thresh_vit_over) / tc_vit
        p_vit_base += 2 * dev_over**2
    total_nutrient_base_penalty += p_vit_base

    # --- Carbs and Iron Base Penalties (example of 'other nutrients') ---
    for nutrient_key in ["carbs", "iron"]:
        ac_nutrient = actual_nutrition_and_cost.get(nutrient_key, 0)
        tc_nutrient = requirements.get(nutrient_key, 1)
        p_nutrient_base = 0
        thresh_lower = tc_nutrient * 0.85
        thresh_upper = tc_nutrient * 1.15
        if not (thresh_lower <= ac_nutrient <= thresh_upper):
            dev = abs(ac_nutrient - tc_nutrient) / tc_nutrient
            p_nutrient_base += 10 * dev**2  # BasePenaltyMult_other_dev
        total_nutrient_base_penalty += p_nutrient_base

    # --- Base Small Portions Penalty P_small_portions_base(x) ---
    # PortionMin_practical = 20g
    small_portion_count = sum(1 for portion in daily_chromosome if 0 < portion < 20)
    p_small_portions_base = 0.75 * small_portion_count

    # 3. Dynamic Penalty Weight PW(Gen)
    # PW(Gen) = 0.5 + 4.5 * (Gen / Gen_max)
    # Ensure max_generations is not zero to avoid division by zero error
    if max_generations == 0:  # Should not happen if GA is set up correctly
        pw_gen = 0.5
    else:
        pw_gen = 0.5 + 4.5 * (generation / max_generations)

    # 4. Fitness Value Used by GA: Fitness_GA(x, Gen)
    # Fitness_GA(x, Gen) = -ActualCost(x) - (PW(Gen) * TotalNutrientBasePenalty(x) + P_small_portions_base(x))
    fitness = -actual_cost - (
        pw_gen * total_nutrient_base_penalty + p_small_portions_base
    )

    # Return fitness and the original actual_nutrition_and_cost (which includes cost now)
    # For tracking purposes, it's good to have the full nutrition breakdown.
    # We re-add cost to actual_nutrition_and_cost for the return, as it was popped.
    actual_nutrition_and_cost_with_cost = actual_nutrition_and_cost.copy()
    actual_nutrition_and_cost_with_cost["cost"] = actual_cost

    return fitness, actual_nutrition_and_cost_with_cost


def tournament_selection(population, fitnesses, tournament_size=250):
    """Select individuals using tournament selection."""
    selected = []

    for _ in range(len(population)):
        # Select tournament_size individuals randomly
        tournament_indices = np.random.choice(
            len(population), tournament_size, replace=False
        )
        tournament_fitnesses = [fitnesses[i] for i in tournament_indices]

        # Select the best individual from the tournament
        winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
        selected.append(population[winner_idx].copy())

    return np.array(selected)


def simulated_binary_crossover(parent1, parent2, eta=4):
    """
    Perform simulated binary crossover between two parents.

    Parameters:
    - parent1, parent2: The two parent solutions
    - eta: Distribution index (higher values keep children closer to parents)

    Returns:
    - child1, child2: The two offspring solutions
    """
    # Copy parents to create children
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Only perform crossover with some probability
    if np.random.random() < 0.9:
        for i in range(len(parent1)):
            # Skip if parents are identical at this gene
            if abs(parent1[i] - parent2[i]) < 1e-10:
                continue

            # Ensure parent1 has the smaller value
            if parent1[i] > parent2[i]:
                parent1[i], parent2[i] = parent2[i], parent1[i]

            # Calculate beta (the crossover parameter)
            rand = np.random.random()
            beta = 1.0 + 2.0 * (parent1[i] - 0) / (parent2[i] - parent1[i])
            alpha = 2.0 - beta ** (-eta - 1)

            if rand <= 1.0 / alpha:
                beta_q = (rand * alpha) ** (1.0 / (eta + 1))
            else:
                beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))

            # Create children using beta_q
            child1[i] = 0.5 * ((1 + beta_q) * parent1[i] + (1 - beta_q) * parent2[i])
            child2[i] = 0.5 * ((1 - beta_q) * parent1[i] + (1 + beta_q) * parent2[i])

            # Ensure children are within bounds
            child1[i] = max(0, min(300, child1[i]))
            child2[i] = max(0, min(300, child2[i]))

    return child1, child2


def gaussian_mutation(individual, mutation_rate=0.6, mutation_scale=25.0):
    """
    Apply Gaussian mutation to an individual.

    Parameters:
    - individual: The solution to mutate
    - mutation_rate: Probability of mutating each gene
    - mutation_scale: Standard deviation of the Gaussian noise
    """
    mutated = individual.copy()

    for i in range(len(mutated)):
        # Apply mutation with probability mutation_rate
        if np.random.random() < mutation_rate:
            # Add Gaussian noise
            mutated[i] += np.random.normal(0, mutation_scale)

            # Ensure the value stays within bounds
            mutated[i] = max(0, min(300, mutated[i]))

    return mutated


def crossover_population(selected, crossover_rate=0.8):
    """Apply crossover to the selected population."""
    offspring = []

    # Ensure we're working with an even number of individuals
    if len(selected) % 2 != 0:
        selected = selected[:-1]

    # Shuffle the population
    np.random.shuffle(selected)

    # Apply crossover to pairs
    for i in range(0, len(selected), 2):
        if np.random.random() < crossover_rate:
            child1, child2 = simulated_binary_crossover(selected[i], selected[i + 1])
            offspring.append(child1)
            offspring.append(child2)
        else:
            offspring.append(selected[i].copy())
            offspring.append(selected[i + 1].copy())

    return np.array(offspring)


def mutate_population(offspring, mutation_rate=0.2):
    """Apply mutation to the offspring population."""
    mutated = []
    for individual in offspring:
        mutated_individual = gaussian_mutation(individual, mutation_rate)
        mutated.append(mutated_individual)
    return np.array(mutated)


def elitism(population, new_population, fitnesses, elite_size=2):
    """
    Preserve the elite_size best individuals from the original population.
    """
    # Get indices of the best individuals
    elite_indices = np.argsort(fitnesses)[-elite_size:]

    # Replace the worst individuals in the new population with the elite from the old population
    for i, idx in enumerate(elite_indices):
        new_population[i] = population[idx].copy()

    return new_population


def genetic_algorithm(user_profile, pop_size=1500, generations=150, elite_size=10):
    """
    Main genetic algorithm loop for DAILY diet planning.
    """
    requirements = calculate_daily_needs(user_profile)
    num_foods = len(FOOD_ITEMS)  # FOOD_ITEMS must be globally defined

    # Initialize population of 1D daily plans
    population = initialize_population(pop_size, num_foods)

    best_fitness = float("-inf")
    best_individual = None  # Will be a 1D daily plan
    best_nutrition_info = None  # Will be a dict of daily nutrition

    for generation in range(generations):
        fitnesses = []
        nutrition_infos = []  # Store full nutrition info for the best individual

        for individual in population:  # individual is a 1D daily plan
            fitness, nutrition_info = calculate_fitness(
                individual, requirements, user_profile, generation, generations
            )
            fitnesses.append(fitness)
            nutrition_infos.append(nutrition_info)

        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
            best_nutrition_info = nutrition_infos[max_fitness_idx]

            print(
                f"Generation {generation}: New best fitness {best_fitness:.2f}, Cost: EGP{best_nutrition_info['cost']:.2f}"
            )

        selected = tournament_selection(
            population, fitnesses
        )  # Assumes tournament_selection returns 1D plans
        offspring = crossover_population(
            selected
        )  # Assumes crossover_population works with 1D plans

        # Mutation rate can still be dynamic if desired, but PW(Gen) handles penalty scaling
        current_mutation_prob = 0.2 * (1 - generation / generations)  # Example
        offspring = mutate_population(
            offspring, mutation_rate=current_mutation_prob
        )  # Assumes mutate_population works with 1D plans

        population = elitism(population, offspring, fitnesses, elite_size)

    return best_individual, best_nutrition_info  # Return daily plan and its nutrition


def format_meal_plan(daily_chromosome, daily_nutrition_info, user_profile):
    """Format the DAILY meal plan for display."""
    result = ["\n===== OPTIMAL DAILY DIET PLAN ====="]

    cost = daily_nutrition_info.get("cost", 0)
    result.append(f"Total Daily Cost: EGP{cost:.2f}")

    result.append("\nDaily Nutritional Profile:")
    for nutrient, value in daily_nutrition_info.items():
        if nutrient != "cost":
            result.append(f"  - {nutrient.capitalize()}: {value:.1f}")

    result.append("\nFoods to Eat:")
    foods_for_day = []
    for i, portion in enumerate(daily_chromosome):
        if portion > 10:  # Only show foods with significant portions
            food_name = FOOD_ITEMS[i]  # FOOD_ITEMS must be globally defined
            food_item_cost = (
                FOOD_DATABASE[food_name]["cost"] * portion / 100.0
            )  # FOOD_DATABASE must be globally defined
            foods_for_day.append(
                f"    - {food_name.replace('_', ' ').title()}: {portion:.0f}g (EGP{food_item_cost:.2f})"
            )
    if foods_for_day:
        result.extend(foods_for_day)
    else:
        result.append("    No significant food portions for this day.")

    return "\n".join(result)


def main():
    # Example user profile
    user_profile = {
        "age": 21,
        "gender": "Male",
        "weight": 75,
        "height": 183,
        "activity_level": "Moderate",
        "goal": "maintain",
        "allergies": [],
        # "monthly_budget": 2500 # Not used in daily planning directly by fitness function
    }

    print("Running genetic algorithm to find optimal DAILY diet plan...")
    # ... (rest of main, calling genetic_algorithm and format_meal_plan) ...
    # ... existing code ...
    # Run the genetic algorithm
    best_daily_individual, best_daily_nutrition = genetic_algorithm(user_profile)

    # Display the result
    if best_daily_individual is not None:
        result_str = format_meal_plan(
            best_daily_individual, best_daily_nutrition, user_profile
        )
        print(result_str)
    else:
        print("No suitable daily meal plan found.")


if __name__ == "__main__":
    main()
