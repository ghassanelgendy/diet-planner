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
    Initialize a population of meal plans.
    Each individual is represented as a list of food portions in grams.
    """
    population = []
    for _ in range(pop_size):
        # Create an individual with random portions (0-300g) for each food
        individual = np.random.uniform(min_portion, max_portion, num_foods)
        population.append(individual)
        print("Individual:", individual)
        print("Individual sum:", np.sum(individual))
    return np.array(population)


def calculate_nutrition(individual):
    """Calculate total nutrition and cost for a meal plan."""
    total = {
        "calories": 0,
        "protein": 0,
        "fats": 0,
        "carbs": 0,
        "vitamins": 0,
        "iron": 0,
        "cholesterol": 0,
        "cost": 0,
    }

    for i, portion in enumerate(individual):
        if portion > 0:  # Only count foods with non-zero portions
            food = FOOD_ITEMS[i]
            food_data = FOOD_DATABASE[food]

            # Calculate nutrition based on portion size (per 100g)
            factor = portion / 100.0
            for nutrient in total.keys():
                if nutrient in food_data:
                    total[nutrient] += food_data[nutrient] * factor
    return total


def calculate_fitness(
    individual, requirements, user_profile, generation, max_generations
):  # Added user_profile
    """
    Calculate the fitness of an individual meal plan.
    Higher fitness is better.
    """
    nutrition = calculate_nutrition(individual)
    goal = user_profile["goal"].lower()

    # Initialize penalty score
    penalty = 0

    # Dynamic penalty weight that increases over generations
    # Start with lenient penalties and gradually increase strictness
    penalty_weight = 0.5 + 4.5 * (generation / max_generations)

    # Penalties for nutrient deficiencies or excesses
    for nutrient, target in requirements.items():
        if nutrient == "cost":  # Skip cost as it's our objective to minimize
            continue

        actual = nutrition.get(nutrient, 0)
        deviation_ratio = 0

        # --- Calorie Penalty (Goal-Specific) ---
        if nutrient == "calories":
            if goal == "lose":
                if (
                    actual > target * 1.02
                ):  # More than 2% over for weight loss is bad (stricter)
                    deviation_ratio = (actual - target * 1.02) / target
                    penalty += (
                        penalty_weight
                        * 100
                        * deviation_ratio**2  # Increased multiplier
                    )  # Higher penalty for excess
                elif actual < target * 0.95:  # More than 5% below (stricter)
                    deviation_ratio = (target * 0.95 - actual) / (target * 0.95)
                    penalty += (
                        penalty_weight * 60 * deviation_ratio**2
                    )  # Increased multiplier
            elif goal == "gain":
                if (
                    actual < target * 0.98
                ):  # More than 2% below for weight gain is bad (stricter)
                    deviation_ratio = (target * 0.98 - actual) / (target * 0.98)
                    penalty += (
                        penalty_weight
                        * 100
                        * deviation_ratio**2  # Increased multiplier
                    )  # Higher penalty for deficit
                elif actual > target * 1.1:  # More than 10% over (stricter)
                    deviation_ratio = (actual - target * 1.1) / target
                    penalty += (
                        penalty_weight * 40 * deviation_ratio**2
                    )  # Increased multiplier
            else:  # maintain
                if (
                    actual < target * 0.95 or actual > target * 1.05
                ):  # 5% deviation (stricter)
                    deviation_ratio = abs(actual - target) / target
                    penalty += (
                        penalty_weight * 80 * deviation_ratio**2
                    )  # Increased multiplier

        # --- Protein Penalty ---
        elif nutrient == "protein":
            # Generally important to meet protein goals, especially for gain/lose
            if actual < target * 0.9:  # More than 10% below
                deviation_ratio = (target * 0.9 - actual) / (target * 0.9)
                penalty += penalty_weight * 20 * deviation_ratio**2
            elif (
                actual > target * 1.3
            ):  # More than 30% above (can be wasteful or strain kidneys in extreme)
                deviation_ratio = (actual - target * 1.3) / target
                penalty += penalty_weight * 5 * deviation_ratio**2

        # --- Fats and Cholesterol Penalty ---
        elif nutrient in ["fats", "cholesterol"]:
            # Being above target is generally worse for these
            if actual > target * 1.15:  # More than 15% above
                deviation_ratio = (actual - target * 1.15) / target
                penalty += penalty_weight * 15 * deviation_ratio**2
            elif (
                actual < target * 0.7 and nutrient == "fats"
            ):  # Fats are essential, don't go too low
                deviation_ratio = (target * 0.7 - actual) / (target * 0.7)
                penalty += penalty_weight * 10 * deviation_ratio**2

        # --- Vitamins Penalty (more lenient due to arbitrary units) ---
        elif nutrient == "vitamins":
            if actual < target * 0.7:  # Wider acceptable range (30% below)
                deviation_ratio = (target * 0.7 - actual) / (target * 0.7)
                penalty += (
                    penalty_weight * 5 * deviation_ratio**2
                )  # Lower penalty multiplier
            elif actual > target * 1.5:  # Wider acceptable range (50% above)
                deviation_ratio = (actual - target * 1.5) / target
                penalty += (
                    penalty_weight * 2 * deviation_ratio**2
                )  # Lower penalty multiplier

        # --- Other Nutrients (Carbs, Iron) ---
        else:
            # Default: Equal penalty for being too high or too low within a 10-15% range
            if actual < target * 0.85 or actual > target * 1.15:  # 15% deviation
                deviation_ratio = abs(actual - target) / target
                penalty += penalty_weight * 10 * deviation_ratio**2

    # Penalty for having too many foods with tiny portions (adds complexity)
    # Consider making 20g a parameter or scaling penalty more
    small_portion_count = sum(1 for p in individual if 0 < p < 20)
    penalty += (
        small_portion_count * 0.75
    )  # Slightly increased penalty for tiny portions

    # Allergy penalty (if implemented)
    # if 'allergies' in user_profile and user_profile['allergies']:
    #     for i, portion in enumerate(individual):
    #         if FOOD_ITEMS[i] in user_profile['allergies'] and portion > 0:
    #             penalty += 1000  # Large penalty for allergens

    # The main objective: minimize cost
    cost = nutrition["cost"]

    # Final fitness: higher is better, so we negate the cost and penalties
    fitness = -cost - penalty

    return fitness, nutrition


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


def genetic_algorithm(user_profile, pop_size=1500, generations=200, elite_size=10):
    """
    Main genetic algorithm loop for diet planning.
    """
    # Calculate nutritional requirements
    requirements = calculate_daily_needs(user_profile)

    # Initialize population
    num_foods = len(FOOD_ITEMS)
    population = initialize_population(pop_size, num_foods)

    best_fitness = float("-inf")
    best_individual = None
    best_nutrition = None

    # Main loop
    for generation in range(generations):
        # Evaluate fitness
        fitnesses = []
        nutritions = []
        for individual in population:
            fitness, nutrition = calculate_fitness(  # Pass user_profile here
                individual, requirements, user_profile, generation, generations
            )
            fitnesses.append(fitness)
            nutritions.append(nutrition)

        # Track best solution
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
            best_nutrition = nutritions[max_fitness_idx]

            print(
                f"Generation {generation}: Found better solution with fitness {best_fitness:.2f}"
            )

        # Selection
        selected = tournament_selection(population, fitnesses)

        # Crossover
        offspring = crossover_population(selected)

        # Mutation
        offspring = mutate_population(offspring)

        # Elitism
        population = elitism(population, offspring, fitnesses, elite_size)

        # Reduce mutation rate over time for fine-tuning
        mutation_rate = 0.2 * (1 - generation / generations)

    # Return best meal plan found
    return best_individual, best_nutrition


def format_meal_plan(individual, nutrition):
    """Format the meal plan for display."""
    result = []
    result.append("\n===== OPTIMAL DIET PLAN =====")

    total_calories = nutrition["calories"]
    total_cost = nutrition["cost"]

    result.append(f"\nTotal Daily Calories: {total_calories:.1f} kcal")
    result.append(f"Total Daily Cost: EGP{total_cost:.2f}")

    result.append("\nNutritional Profile:")
    result.append(f"  - Protein: {nutrition['protein']:.1f}g")
    result.append(f"  - Fats: {nutrition['fats']:.1f}g")
    result.append(f"  - Carbs: {nutrition['carbs']:.1f}g")
    result.append(f"  - Iron: {nutrition['iron']:.1f}mg")
    result.append(f"  - Cholesterol: {nutrition['cholesterol']:.1f}mg")

    result.append("\nRecommended Daily Foods:")
    for i, portion in enumerate(individual):
        if portion > 10:  # Only show foods with significant portions
            food = FOOD_ITEMS[i]
            food_cost = FOOD_DATABASE[food]["cost"] * portion / 100.0
            result.append(
                f"  - {food.replace('_', ' ').title()}: {portion:.0f}g (EGP{food_cost:.2f})"
            )

    return "\n".join(result)


def main():
    # Example user profile
    user_profile = {
        "age": 21,
        "gender": "Male",
        "weight": 75,  # kg
        "height": 183,  # cm
        "activity_level": "Moderate",
        "goal": "a88",
        "allergies": [],
    }

    print("Running genetic algorithm to find optimal diet plan...")
    print("This may take a few minutes...")

    # Run the genetic algorithm
    best_individual, best_nutrition = genetic_algorithm(user_profile)

    # Display the result
    result = format_meal_plan(best_individual, best_nutrition)
    print(result)


if __name__ == "__main__":
    main()
