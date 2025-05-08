import numpy as np
import pandas as pd
from typing import Dict, Any
from food_database import FOOD_DATABASE, FOOD_ITEMS, FOOD_GROUPS


def calculate_daily_needs(user_profile: Dict[str, Any]) -> Dict[str, float]:
    """Calculate daily nutritional requirements based on user profile."""
    age = user_profile["age"]
    gender = user_profile["gender"].lower()
    weight = user_profile["weight"]  # kg
    height = user_profile["height"]  # cm
    activity_level = user_profile["activity_level"].lower()
    goal = user_profile["goal"].lower()

    # Calculate BMR (Basal Metabolic Rate) using Mifflin-St Jeor equation
    bmr = 10 * weight + 6.25 * height - 5 * age + (5 if gender == "male" else -161)

    # Activity level multipliers
    activity_multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very active": 1.9,
    }

    # Calculate TDEE (Total Daily Energy Expenditure)
    tdee = bmr * activity_multipliers[activity_level]

    # Adjust calories based on goal
    if goal == "lose":
        calories = tdee - 500  # 500 calorie deficit for weight loss
    elif goal == "gain":
        calories = tdee + 500  # 500 calorie surplus for weight gain
    else:  # maintain
        calories = tdee

    # Calculate macronutrient needs
    if goal == "gain":
        protein_ratio, fat_ratio, carb_ratio = 0.15, 0.35, 0.5
    elif goal == "lose":
        protein_ratio, fat_ratio, carb_ratio = 0.25, 0.25, 0.5
    else:  # maintain
        protein_ratio, fat_ratio, carb_ratio = 0.35, 0.20, 0.45

    protein_calories = calories * protein_ratio
    fat_calories = calories * fat_ratio
    carb_calories = calories * carb_ratio

    protein_grams = protein_calories / 4  # 4 calories per gram of protein
    fat_grams = fat_calories / 9  # 9 calories per gram of fat
    carb_grams = carb_calories / 4  # 4 calories per gram of carbs

    # Other nutrient needs (simplified estimates)
    if gender == "male" or age > 50:
        iron_mg = 8
        cholesterol_max = 300  # mg
    else:
        iron_mg = 18
        cholesterol_max = 300  # mg

    return {
        "calories": calories,
        "protein": protein_grams,
        "fats": fat_grams,
        "carbs": carb_grams,
        "iron": iron_mg,
        "cholesterol": cholesterol_max,
    }


def initialize_population(pop_size, num_foods, min_portion=0, max_portion=300):
    """
    Initialize a population of daily meal plans.
    Each individual is a 1D array of food portions in grams.
    Shape: (pop_size, num_foods)
    """
    return np.random.uniform(min_portion, max_portion, (pop_size, num_foods))


def calculate_nutrition_and_cost_for_day(daily_chromosome_portions):
    """Calculate total nutrition and cost for a single day's meal plan (1D chromosome)."""
    totals = {k: 0 for k in ["calories", "protein", "fats", "carbs", "iron", "cholesterol", "cost"]}
    for i, portion in enumerate(daily_chromosome_portions):
        if portion > 0:
            food = FOOD_ITEMS[i]
            data = FOOD_DATABASE[food]
            factor = portion / 100.0
            for k in totals:
                if k in data:
                    totals[k] += data[k] * factor
    return totals


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
    actual = calculate_nutrition_and_cost_for_day(daily_chromosome)
    actual_cost = actual.pop(
        "cost"
    )  # Remove cost for nutrient penalty calculation

    # 2. Calculate Base Penalty Functions (Unscaled)
    total_penalty = 0
    goal = user_profile["goal"].lower()  # GoalUser

    # Define BasePenaltyMult_k_condition and Thresh_k_condition (as per your previous script logic)
    # These are the multipliers and thresholds for penalties BEFORE PW(Gen) scaling.
    # Example for calories (you'll need to define these for all nutrients based on your old penalty structure)
    # This section needs to mirror the penalty logic from your original daily planner's fitness function.

    # --- Calorie Base Penalty P_calories_base(x) ---
    ac_calories = actual.get("calories", 0)
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
    total_penalty += p_calories_base

    # --- Protein Base Penalty P_protein_base(x) ---
    ac_protein = actual.get("protein", 0)
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
    total_penalty += p_protein_base

    # --- Fats Base Penalty P_fats_base(x) ---
    ac_fats = actual.get("fats", 0)
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
    total_penalty += p_fats_base

    # --- Cholesterol Base Penalty P_cholesterol_base(x) ---
    ac_chol = actual.get("cholesterol", 0)
    tc_chol = requirements.get("cholesterol", 1)  # Max target
    p_chol_base = 0
    thresh_chol_over = tc_chol * 1.15  # Max is 1.0 * target, so 1.15 is over
    if ac_chol > thresh_chol_over:  # Only penalize if over
        dev_over = (
            ac_chol - thresh_chol_over
        ) / tc_chol  # Denominator could be tc_chol or thresh_chol_over
        p_chol_base += 15 * dev_over**2  # Using same multiplier as fats for "over"
    total_penalty += p_chol_base

    # --- Carbs and Iron Base Penalties (example of 'other nutrients') ---
    for nutrient_key in ["carbs", "iron"]:
        ac_nutrient = actual.get(nutrient_key, 0)
        tc_nutrient = requirements.get(nutrient_key, 1)
        p_nutrient_base = 0
        thresh_lower = tc_nutrient * 0.85
        thresh_upper = tc_nutrient * 1.15
        if not (thresh_lower <= ac_nutrient <= thresh_upper):
            dev = abs(ac_nutrient - tc_nutrient) / tc_nutrient
            p_nutrient_base += 10 * dev**2  # BasePenaltyMult_other_dev
        total_penalty += p_nutrient_base

    # --- Diversity Penalty: All Food Groups ---
    group_counts = {}
    for i, portion in enumerate(daily_chromosome):
        if portion > 10:  # Only count significant portions
            food = FOOD_ITEMS[i]
            group = FOOD_GROUPS.get(food, None)
            if group:
                group_counts[group] = group_counts.get(group, 0) + 1
    diversity_penalty = sum(50 * (count - 1) for count in group_counts.values() if count > 1)

    # --- Base Small Portions Penalty P_small_portions_base(x) ---
    # PortionMin_practical = 20g
    small_portion_count = sum(1 for portion in daily_chromosome if 0 < portion < 20)
    small_portion_penalty = 0.75 * small_portion_count

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
        pw_gen * total_penalty + small_portion_penalty
    )
    fitness -= diversity_penalty

    # Return fitness and the original actual (which includes cost now)
    # For tracking purposes, it's good to have the full nutrition breakdown.
    # We re-add cost to actual for the return, as it was popped.
    actual_with_cost = actual.copy()
    actual_with_cost["cost"] = actual_cost

    return fitness, actual_with_cost


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


def genetic_algorithm(user_profile, pop_size=1500, generations=20, elite_size=10):
    """
    Main genetic algorithm loop for DAILY diet planning.
    """
    requirements = calculate_daily_needs(user_profile)
    num_foods = len(FOOD_ITEMS)
    population = initialize_population(pop_size, num_foods)
    best_fitness = float("-inf")
    best_individual = None
    best_nutrition_info = None
    for generation in range(generations):
        fitnesses = []
        nutrition_infos = []
        for individual in population:
            fitness, nutrition_info = calculate_fitness(individual, requirements, user_profile, generation, generations)
            fitnesses.append(fitness)
            nutrition_infos.append(nutrition_info)
        max_fitness_idx = np.argmax(fitnesses)
        if fitnesses[max_fitness_idx] > best_fitness:
            best_fitness = fitnesses[max_fitness_idx]
            best_individual = population[max_fitness_idx].copy()
            best_nutrition_info = nutrition_infos[max_fitness_idx]
        # Print progress for this generation
        print(f"Generation {generation+1}/{generations}: Best Fitness = {fitnesses[max_fitness_idx]:.2f}, Cost = EGP{nutrition_infos[max_fitness_idx]['cost']:.2f}")
        selected = tournament_selection(population, fitnesses)
        offspring = crossover_population(selected)
        current_mutation_prob = 0.2 * (1 - generation / generations)
        offspring = mutate_population(offspring, mutation_rate=current_mutation_prob)
        population = elitism(population, offspring, fitnesses, elite_size)
    return best_individual, best_nutrition_info


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
        if portion > 10:
            food_name = FOOD_ITEMS[i]
            food_item_cost = FOOD_DATABASE[food_name]["cost"] * portion / 100.0
            foods_for_day.append(f"    - {food_name.replace('_', ' ').title()}: {portion:.0f}g (EGP{food_item_cost:.2f})")
    if foods_for_day:
        result.extend(foods_for_day)
    else:
        result.append("    No significant food portions for this day.")
    return "\n".join(result)


def weekly_genetic_algorithm(user_profiles, pop_size=1500, generations=20, elite_size=10, randomize_seed=True):
    """
    Run the daily genetic algorithm 7 times, tracking foods used and penalizing repeats for variety.
    user_profiles: list of 7 user_profile dicts (can be the same or different for each day)
    Returns: list of (daily_chromosome, daily_nutrition_info) for each day, and a weekly nutrition summary.
    """
    days = 7
    weekly_food_counts = np.zeros(len(FOOD_ITEMS))
    weekly_nutrition = {k: 0 for k in ["calories", "protein", "fats", "carbs", "iron", "cholesterol", "cost"]}
    daily_results = []

    def fitness_with_weekly_penalty(daily_chromosome, requirements, user_profile, generation, max_generations, food_counts):
        # Use the original fitness function
        fitness, nutrition_info = calculate_fitness(
            daily_chromosome, requirements, user_profile, generation, max_generations
        )
        # Add penalty for foods already used a lot this week
        penalty = 0
        for i, portion in enumerate(daily_chromosome):
            if portion > 10:
                penalty += 20 * food_counts[i]  # 20 points per previous use
        fitness -= penalty
        return fitness, nutrition_info

    for day in range(days):
        if randomize_seed:
            np.random.seed(None)
        user_profile = user_profiles[day]
        requirements = calculate_daily_needs(user_profile)
        num_foods = len(FOOD_ITEMS)
        population = initialize_population(pop_size, num_foods)
        best_fitness = float("-inf")
        best_individual = None
        best_nutrition_info = None
        for generation in range(generations):
            fitnesses = []
            nutrition_infos = []
            for individual in population:
                fitness, nutrition_info = fitness_with_weekly_penalty(
                    individual, requirements, user_profile, generation, generations, weekly_food_counts
                )
                fitnesses.append(fitness)
                nutrition_infos.append(nutrition_info)
            max_fitness_idx = np.argmax(fitnesses)
            if fitnesses[max_fitness_idx] > best_fitness:
                best_fitness = fitnesses[max_fitness_idx]
                best_individual = population[max_fitness_idx].copy()
                best_nutrition_info = nutrition_infos[max_fitness_idx]
            print(f"[Day {day+1}] Generation {generation+1}/{generations}: Best Fitness = {fitnesses[max_fitness_idx]:.2f}, Cost = EGP{nutrition_infos[max_fitness_idx]['cost']:.2f}")
            selected = tournament_selection(population, fitnesses)
            offspring = crossover_population(selected)
            current_mutation_prob = 0.2 * (1 - generation / generations)
            offspring = mutate_population(offspring, mutation_rate=current_mutation_prob)
            population = elitism(population, offspring, fitnesses, elite_size)
        # Track foods used in significant portions
        for i, portion in enumerate(best_individual):
            if portion > 10:
                weekly_food_counts[i] += 1
        # Track weekly nutrition
        for k in weekly_nutrition:
            weekly_nutrition[k] += best_nutrition_info.get(k, 0)
        daily_results.append((best_individual, best_nutrition_info))
    # Compute weekly nutrition averages
    weekly_averages = {k: v / days for k, v in weekly_nutrition.items()}
    return daily_results, weekly_nutrition, weekly_averages


def get_weekly_user_profiles(base_profile):
    """
    Ask user if they want to set different goals for different days.
    Returns a list of 7 user_profile dicts.
    """
    print("\nDo you want to set different goals for different days? (y/N)")
    ans = input().strip().lower()
    if ans == "y":
        profiles = []
        for i in range(7):
            print(f"\nDay {i+1}:")
            print("  (M)aintain, (L)ose, (G)ain")
            goal_input = input("  Goal: ").strip().lower()[:1]
            goal_map = {"m": "maintain", "l": "lose", "g": "gain"}
            goal = goal_map.get(goal_input, base_profile["goal"])
            day_profile = base_profile.copy()
            day_profile["goal"] = goal
            profiles.append(day_profile)
        return profiles
    else:
        return [base_profile.copy() for _ in range(7)]


def format_weekly_plan(daily_results, weekly_nutrition, weekly_averages):
    result = ["\n===== OPTIMAL WEEKLY DIET PLAN ====="]
    total_cost = weekly_nutrition.get("cost", 0)
    result.append(f"Total Weekly Cost: EGP{total_cost:.2f}")
    result.append("\nAverage Daily Nutrition:")
    for nutrient, value in weekly_averages.items():
        if nutrient != "cost":
            result.append(f"  - {nutrient.capitalize()}: {value:.1f}")
    for day, (chromosome, nutrition) in enumerate(daily_results, 1):
        result.append(f"\n--- Day {day} ---")
        result.append(format_meal_plan(chromosome, nutrition, {}))
    return "\n".join(result)


def main_menu():
    print("\n==== Diet Planner Main Menu ====")
    print("1. Run with example profile")
    print("2. Enter custom profile")
    print("3. Run weekly planner")
    print("0. Exit")
    choice = input("Select an option: ").strip()
    return choice[:1]  # Only first character


def get_user_profile():
    print("\nEnter your profile information:")
    age = int(input("Age: "))
    gender_input = input("Gender (M/F): ").strip().lower()[:1]
    gender = "Male" if gender_input == "m" else "Female"
    weight = float(input("Weight (kg): "))
    height = float(input("Height (cm): "))
    activity_map = {
        "s": "Sedentary",
        "l": "Light",
        "m": "Moderate",
        "a": "Active",
        "v": "Very active"
    }
    activity_input = input("Activity level (S)edentary, (L)ight, (M)oderate, (A)ctive, (V)ery active: ").strip().lower()[:1]
    activity_level = activity_map.get(activity_input, "Moderate")
    goal_map = {"m": "maintain", "l": "lose", "g": "gain"}
    goal_input = input("Goal (M)aintain, (L)ose, (G)ain: ").strip().lower()[:1]
    goal = goal_map.get(goal_input, "maintain")
    return {
        "age": age,
        "gender": gender,
        "weight": weight,
        "height": height,
        "activity_level": activity_level,
        "goal": goal,
    }


def main():
    while True:
        choice = main_menu()
        if choice == "1":
            user_profile = {
                "age": 21,
                "gender": "Male",
                "weight": 75,
                "height": 183,
                "activity_level": "Moderate",
                "goal": "maintain",
            }
            print("\nUsing default example profile:")
            for k, v in user_profile.items():
                print(f"  {k}: {v}")
        elif choice == "2":
            user_profile = get_user_profile()
        elif choice == "3":
            user_profile = get_user_profile()
            weekly_profiles = get_weekly_user_profiles(user_profile)
            print("\nRunning genetic algorithm to find optimal weekly diet plan...")
            daily_results, weekly_nutrition, weekly_averages = weekly_genetic_algorithm(weekly_profiles)
            result_str = format_weekly_plan(daily_results, weekly_nutrition, weekly_averages)
            print(result_str)
            input("\nPress Enter to return to the main menu...")
            continue
        elif choice == "0":
            print("Salam")
            return
        else:
            print("Invalid.")
            continue

        print("\nRunning genetic algorithm to find optimal daily diet plan...")
        best_daily_individual, best_daily_nutrition = genetic_algorithm(user_profile)
        if best_daily_individual is not None:
            result_str = format_meal_plan(best_daily_individual, best_daily_nutrition, user_profile)
            print(result_str)
        else:
            print("No suitable daily meal plan found.")
        input("\nPress Enter to return to the main menu...")


if __name__ == "__main__":
    main()