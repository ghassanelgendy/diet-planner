import numpy as np
import random
import pandas as pd
from typing import Dict, List, Tuple, Any

# Food database with nutrition info and costs
# Each food item has: calories, protein, fats, carbs, vitamins, iron, cholesterol, cost per 100g
FOOD_DATABASE = {
    "chicken_breast": {
        "calories": 165,
        "protein": 31,
        "fats": 3.6,
        "carbs": 0,
        "vitamins": 0.3,
        "iron": 0.9,
        "cholesterol": 85,
        "cost": 30.0,
    },  # ~EGP 300/kg
    "brown_rice": {
        "calories": 112,
        "protein": 2.6,
        "fats": 0.9,
        "carbs": 23,
        "vitamins": 0.2,
        "iron": 0.8,
        "cholesterol": 0,
        "cost": 1.8,
    },  # ~EGP 18/kg
    "white_rice": {
        "calories": 130,
        "protein": 2.7,
        "fats": 0.3,
        "carbs": 28,
        "vitamins": 0.1,
        "iron": 0.2,
        "cholesterol": 0,
        "cost": 1.5,
    },  # ~EGP 15/kg
    "pasta": {
        "calories": 131,
        "protein": 5,
        "fats": 1.1,
        "carbs": 25,
        "vitamins": 0.1,
        "iron": 0.5,
        "cholesterol": 0,
        "cost": 1.2,
    },  # ~EGP 12/kg
    "fava_beans": {
        "calories": 110,
        "protein": 7.6,
        "fats": 0.5,
        "carbs": 19,
        "vitamins": 0.2,
        "iron": 1.5,
        "cholesterol": 0,
        "cost": 1.0,
    },  # ~EGP 10/kg
    "baladi_bread": {
        "calories": 260,
        "protein": 8,
        "fats": 1,
        "carbs": 52,
        "vitamins": 0.3,
        "iron": 1.4,
        "cholesterol": 0,
        "cost": 0.5,
    },  # ~EGP 5/kg
    "tilapia_fish": {
        "calories": 128,
        "protein": 26,
        "fats": 2.7,
        "carbs": 0,
        "vitamins": 0.4,
        "iron": 0.6,
        "cholesterol": 50,
        "cost": 18.0,
    },  # ~EGP 180/kg
    "broccoli": {
        "calories": 34,
        "protein": 2.8,
        "fats": 0.4,
        "carbs": 7,
        "vitamins": 1.5,
        "iron": 0.7,
        "cholesterol": 0,
        "cost": 4.0,
    },  # ~EGP 40/kg
    "sweet_potato": {
        "calories": 86,
        "protein": 1.6,
        "fats": 0.1,
        "carbs": 20,
        "vitamins": 1.3,
        "iron": 0.6,
        "cholesterol": 0,
        "cost": 1.5,
    },  # ~EGP 15/kg
    "eggs": {
        "calories": 155,
        "protein": 13,
        "fats": 11,
        "carbs": 1.1,
        "vitamins": 0.9,
        "iron": 1.8,
        "cholesterol": 186,
        "cost": 1.7,
    },  # ~EGP 17/dozen (~100g)
    "spinach": {
        "calories": 23,
        "protein": 2.9,
        "fats": 0.4,
        "carbs": 3.6,
        "vitamins": 1.9,
        "iron": 2.7,
        "cholesterol": 0,
        "cost": 2.0,
    },  # ~EGP 20/kg
    "molokhia": {
        "calories": 32,
        "protein": 3.7,
        "fats": 0.3,
        "carbs": 7.1,
        "vitamins": 2.0,
        "iron": 3.2,
        "cholesterol": 0,
        "cost": 1.5,
    },  # ~EGP 15/kg
    "okra": {
        "calories": 33,
        "protein": 2.0,
        "fats": 0.1,
        "carbs": 7.5,
        "vitamins": 0.9,
        "iron": 0.8,
        "cholesterol": 0,
        "cost": 2.0,
    },  # ~EGP 20/kg
    "eggplant": {
        "calories": 25,
        "protein": 1.0,
        "fats": 0.2,
        "carbs": 6,
        "vitamins": 0.5,
        "iron": 0.4,
        "cholesterol": 0,
        "cost": 1.0,
    },  # ~EGP 10/kg
    "zabadi_yogurt": {
        "calories": 63,
        "protein": 5.3,
        "fats": 3.3,
        "carbs": 4.7,
        "vitamins": 0.3,
        "iron": 0.1,
        "cholesterol": 13,
        "cost": 2.5,
    },  # ~EGP 25/kg
    "white_cheese": {
        "calories": 264,
        "protein": 14,
        "fats": 21,
        "carbs": 4.1,
        "vitamins": 0.5,
        "iron": 0.7,
        "cholesterol": 90,
        "cost": 4.5,
    },  # ~EGP 45/kg
    "banana": {
        "calories": 89,
        "protein": 1.1,
        "fats": 0.3,
        "carbs": 23,
        "vitamins": 0.7,
        "iron": 0.3,
        "cholesterol": 0,
        "cost": 2.0,
    },  # ~EGP 20/kg
    "dates": {
        "calories": 282,
        "protein": 2.5,
        "fats": 0.4,
        "carbs": 75,
        "vitamins": 0.5,
        "iron": 1.0,
        "cholesterol": 0,
        "cost": 3.0,
    },  # ~EGP 30/kg
    "oranges": {
        "calories": 47,
        "protein": 0.9,
        "fats": 0.1,
        "carbs": 12,
        "vitamins": 1.5,
        "iron": 0.1,
        "cholesterol": 0,
        "cost": 1.5,
    },  # ~EGP 15/kg
    "almonds": {
        "calories": 576,
        "protein": 21,
        "fats": 49,
        "carbs": 22,
        "vitamins": 0.6,
        "iron": 3.7,
        "cholesterol": 0,
        "cost": 20.0,
    },  # ~EGP 200/kg
    "oats": {
        "calories": 389,
        "protein": 16.9,
        "fats": 6.9,
        "carbs": 66.3,
        "vitamins": 0.5,
        "iron": 4.7,
        "cholesterol": 0,
        "cost": 5.0,
    },  # ~EGP 50/kg
    "beef": {
        "calories": 250,
        "protein": 26,
        "fats": 17,
        "carbs": 0,
        "vitamins": 0.4,
        "iron": 2.6,
        "cholesterol": 70,
        "cost": 27.0,
    },  # ~EGP 270/kg
    "lentils": {
        "calories": 116,
        "protein": 9,
        "fats": 0.4,
        "carbs": 20,
        "vitamins": 0.3,
        "iron": 3.3,
        "cholesterol": 0,
        "cost": 3.0,
    },  # ~EGP 30/kg
    "tahini": {
        "calories": 595,
        "protein": 17,
        "fats": 53,
        "carbs": 21,
        "vitamins": 0.2,
        "iron": 9.2,
        "cholesterol": 0,
        "cost": 6.0,
    },  # ~EGP 60/kg
    "olive_oil": {
        "calories": 884,
        "protein": 0,
        "fats": 100,
        "carbs": 0,
        "vitamins": 1.0,
        "iron": 0.1,
        "cholesterol": 0,
        "cost": 15.0,
    },  # ~EGP 150/liter
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


def calculate_fitness(individual, requirements, generation, max_generations):
    """
    Calculate the fitness of an individual meal plan.
    Higher fitness is better.
    """
    nutrition = calculate_nutrition(individual)

    # Initialize penalty score
    penalty = 0

    # Dynamic penalty weight that increases over generations
    # Start with lenient penalties and gradually increase strictness
    penalty_weight = 0.5 + 4.5 * (generation / max_generations)

    # Penalties for nutrient deficiencies or excesses
    # We want to be within 10% of the target for each nutrient
    for nutrient, target in requirements.items():
        if nutrient != "cost":  # Skip cost as it's our objective to minimize
            actual = nutrition.get(nutrient, 0)

            # Different penalties for deficiencies vs. excesses
            if nutrient in ["calories", "protein", "vitamins", "iron"]:
                # Being below target is worse than being above for these
                if actual < target * 0.9:  # More than 10% below
                    deficit_ratio = (target * 0.9 - actual) / (target * 0.9)
                    penalty += penalty_weight * 20 * deficit_ratio**2
                elif actual > target * 1.2:  # More than 20% above
                    excess_ratio = (actual - target * 1.2) / target
                    penalty += penalty_weight * 5 * excess_ratio**2
            elif nutrient in ["fats", "cholesterol"]:
                # Being above target is worse for these
                if actual > target * 1.1:  # More than 10% above
                    excess_ratio = (actual - target * 1.1) / target
                    penalty += penalty_weight * 10 * excess_ratio**2
                elif actual < target * 0.7:  # More than 30% below
                    deficit_ratio = (target * 0.7 - actual) / (target * 0.7)
                    penalty += penalty_weight * 5 * deficit_ratio**2
            else:  # carbs and others
                # Equal penalty for being too high or too low
                if actual < target * 0.9 or actual > target * 1.1:
                    deviation = abs(actual - target) / target
                    penalty += penalty_weight * 10 * deviation**2

    # Penalty for having too many foods with tiny portions (adds complexity)
    small_portion_count = sum(1 for p in individual if 0 < p < 20)
    penalty += small_portion_count * 0.5

    # Check for allergies (would be implemented if allergies were provided)
    # for i, portion in enumerate(individual):
    #     if FOOD_ITEMS[i] in user_profile['allergies'] and portion > 0:
    #         penalty += 1000  # Large penalty for allergens

    # The main objective: minimize cost
    cost = nutrition["cost"]

    # Final fitness: higher is better, so we negate the cost and penalties
    fitness = -cost - penalty

    return fitness, nutrition


def tournament_selection(population, fitnesses, tournament_size=7):
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
            fitness, nutrition = calculate_fitness(
                individual, requirements, generation, generations
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
