import numpy as np
import pandas as pd
from typing import Dict, Any
from food_database import FOOD_DATABASE, FOOD_ITEMS


def calculate_daily_needs(user_profile: Dict[str, Any]) -> Dict[str, float]:
    """Calculate daily nutritional requirements based on user profile."""
    age = user_profile["age"]
    gender = user_profile["gender"].lower()
    weight = user_profile["weight"]  # kg
    height = user_profile["height"]  # cm
    activity_level = user_profile["activity_level"].lower()
    goal = user_profile["goal"].lower()

    # Hesbet elBMR (Basal Metabolic Rate)
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

    # cholesterol w iron needs
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


def initialize_population(
    pop_size, num_foods, min_portion=0, max_portion=300
):  # daily meal plan bs random
    return np.random.uniform(min_portion, max_portion, (pop_size, num_foods))


def calculate_nutrition_and_cost_for_day(
    daily_chromosome_portions,
):  # matrix feh cost elmeals w nutration facts
    totals = {
        k: 0
        for k in ["calories", "protein", "fats", "carbs", "iron", "cholesterol", "cost"]
    }
    for i, portion in enumerate(daily_chromosome_portions):
        if portion > 0:
            food = FOOD_ITEMS[i]
            data = FOOD_DATABASE[food]
            factor = portion / 100.0  # kool elakl feh nutration facts fel 100gram
            for k in totals:
                if k in data:
                    totals[k] += data[k] * factor
    return totals


def trapezoidal_membership(x, a, b, c, d):
    """
    Trapezoidal membership function for fuzzy logic.
    Returns a value between 0 and 1.
    a, d: where membership is 0
    b, c: where membership is 1
    """
    if x <= a or x >= d:
        return 0.0
    elif b <= x <= c:
        return 1.0
    elif a < x < b:
        return (x - a) / (b - a)
    elif c < x < d:
        return (d - x) / (d - c)
    else:
        return 0.0


def calculate_fitness(
    meal_plan_chromosome,
    nutrient_requirements,
    user_profile,
    current_generation,
    total_generations,
):

    # Fitness(x, Gen) = -ActualCost(x) - (PW(Gen) * TotalNutrientBasePenalty(x) + P_small_portions_base(x))
    # 1. Calculate Actual_k(x) and ActualCost(x)
    actual_nutrients = calculate_nutrition_and_cost_for_day(meal_plan_chromosome)
    actual_total_cost = actual_nutrients.pop("cost")

    total_nutrient_penalty = 0
    user_goal = user_profile["goal"].lower()

    # --- Fuzzy Calorie Compliance ---
    actual_calories = actual_nutrients.get("calories", 0)
    target_calories = nutrient_requirements.get("calories", 1)
    # Define fuzzy "acceptable" region (customize as needed)
    a = target_calories * 0.90  # 0 membership below this
    b = target_calories * 0.97  # 1 membership above this
    c = target_calories * 1.03  # 1 membership below this
    d = target_calories * 1.10  # 0 membership above this
    calorie_fuzzy_score = trapezoidal_membership(actual_calories, a, b, c, d)
    # Use (1 - calorie_fuzzy_score) as a soft penalty, scaled
    fuzzy_calorie_penalty = (1 - calorie_fuzzy_score) * 100  # scale as needed
    total_nutrient_penalty += fuzzy_calorie_penalty

    # --- Protein Base Penalty penalty_protein ---
    actual_protein = actual_nutrients.get("protein", 0)
    target_protein = nutrient_requirements.get("protein", 1)
    penalty_protein = 0
    thresh_prot_under = target_protein * 0.90
    thresh_prot_over = target_protein * 1.30
    if actual_protein < thresh_prot_under:
        dev_under = (thresh_prot_under - actual_protein) / thresh_prot_under
        penalty_protein += 20 * dev_under**2
    elif actual_protein > thresh_prot_over:
        dev_over = (actual_protein - thresh_prot_over) / target_protein
        penalty_protein += 5 * dev_over**2
    total_nutrient_penalty += penalty_protein

    # --- Fats Base Penalty penalty_fats ---
    actual_fats = actual_nutrients.get("fats", 0)
    target_fats = nutrient_requirements.get("fats", 1)
    penalty_fats = 0
    thresh_fats_over = target_fats * 1.15
    thresh_fats_under = target_fats * 0.70
    if actual_fats > thresh_fats_over:
        dev_over = (actual_fats - thresh_fats_over) / target_fats
        penalty_fats += 15 * dev_over**2
    elif actual_fats < thresh_fats_under:
        dev_under = (thresh_fats_under - actual_fats) / thresh_fats_under
        penalty_fats += 10 * dev_under**2
    total_nutrient_penalty += penalty_fats

    # --- Cholesterol Base Penalty penalty_cholesterol ---
    actual_cholesterol = actual_nutrients.get("cholesterol", 0)
    target_cholesterol = nutrient_requirements.get("cholesterol", 1)
    penalty_cholesterol = 0
    thresh_chol_over = target_cholesterol * 1.15
    if actual_cholesterol > thresh_chol_over:
        dev_over = (actual_cholesterol - thresh_chol_over) / target_cholesterol
        penalty_cholesterol += 15 * dev_over**2
    total_nutrient_penalty += penalty_cholesterol

    # --- Carbs and Iron Base Penalties (example of 'other nutrients') ---
    for nutrient_key in ["carbs", "iron"]:
        actual_nutrient = actual_nutrients.get(nutrient_key, 0)
        target_nutrient = nutrient_requirements.get(nutrient_key, 1)
        penalty_other_nutrient = 0
        thresh_lower = target_nutrient * 0.85
        thresh_upper = target_nutrient * 1.15
        if not (thresh_lower <= actual_nutrient <= thresh_upper):
            dev = abs(actual_nutrient - target_nutrient) / target_nutrient
            penalty_other_nutrient += 10 * dev**2
        total_nutrient_penalty += penalty_other_nutrient

    # --- Base Small Portions Penalty small_portions_penalty ---
    small_portions_count = sum(
        1 for portion in meal_plan_chromosome if 0 < portion < 20
    )
    small_portions_penalty = 0.75 * small_portions_count

    # 3. Dynamic Penalty Weight penalty_weight
    if total_generations == 0:
        penalty_weight = 0.5
    else:
        penalty_weight = 0.5 + 4.5 * (current_generation / total_generations)

    # 4. Fitness Value Used by GA: Fitness_GA(x, Gen)
    fitness = -actual_total_cost - (
        penalty_weight * total_nutrient_penalty + small_portions_penalty
    )

    # Return fitness and the original actual (which includes cost now)
    actual_nutrients_with_cost = actual_nutrients.copy()
    actual_nutrients_with_cost["cost"] = actual_total_cost

    return fitness, actual_nutrients_with_cost


def tournament_selection(population, fitnesses, tournament_size=20):
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


def gaussian_mutation(individual, mutation_rate=0.15, mutation_scale=20):
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


def mutate_population(offspring, mutation_rate=0.05):
    """Apply mutation to the offspring population."""
    mutated = []
    for individual in offspring:
        mutated_individual = gaussian_mutation(individual, mutation_rate)
        mutated.append(mutated_individual)
    return np.array(mutated)


def elitism(population, new_population, fitnesses, elite_size):
    """
    Preserve the elite_size best individuals from the original population.
    """
    # Get indices of the best individuals
    elite_indices = np.argsort(fitnesses)[-elite_size:]

    # Replace the worst individuals in the new population with the elite from the old population
    for i, idx in enumerate(elite_indices):
        new_population[i] = population[idx].copy()

    return new_population


def genetic_algorithm(user_profile, pop_size=400, generations=60, elite_size=15):
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
        # Print progress for this generation
        print(
            f"Generation {generation+1}/{generations}: Best Fitness = {fitnesses[max_fitness_idx]:.2f}, Cost = EGP{nutrition_infos[max_fitness_idx]['cost']:.2f}"
        )
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
            foods_for_day.append(
                f"    - {food_name.replace('_', ' ').title()}: {portion:.0f}g (EGP{food_item_cost:.2f})"
            )
    if foods_for_day:
        result.extend(foods_for_day)
    else:
        result.append("    No significant food portions for this day.")
    return "\n".join(result)


def generate_weekly_shopping_list(daily_results):
    food_totals = {food: 0.0 for food in FOOD_ITEMS}
    food_costs = {food: 0.0 for food in FOOD_ITEMS}
    for chromosome, _ in daily_results:
        for i, portion in enumerate(chromosome):
            if portion > 0:
                food = FOOD_ITEMS[i]
                food_totals[food] += portion
                food_costs[food] += FOOD_DATABASE[food]["cost"] * portion / 100.0
    shopping_list = []
    for food in FOOD_ITEMS:
        if food_totals[food] > 0:
            shopping_list.append((food, food_totals[food], food_costs[food]))
    shopping_list.sort(key=lambda x: -x[1])
    return shopping_list


def format_shopping_list(shopping_list):
    result = ["\n===== WEEKLY SHOPPING LIST ====="]
    total_cost = sum(item[2] for item in shopping_list)
    for food, grams, cost in shopping_list:
        result.append(
            f"- {food.replace('_', ' ').title()}: {grams:.0f}g (EGP{cost:.2f})"
        )
    result.append(f"\nTotal Shopping Cost: EGP{total_cost:.2f}")
    return "\n".join(result)


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


def format_weekly_plan(daily_results, weekly_nutrition, weekly_averages, shopping_list):
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
    result.append(format_shopping_list(shopping_list))
    return "\n".join(result)


def main_menu():
    print("\n==== Diet Planner Main Menu ====")
    print("1. Run with example profile")
    print("2. Enter custom profile")
    print("3. Run weekly planner (custom profile)")
    print("4. Run weekly planner (example profile)")
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
        "v": "Very active",
    }
    activity_input = (
        input(
            "Activity level (S)edentary, (L)ight, (M)oderate, (A)ctive, (V)ery active: "
        )
        .strip()
        .lower()[:1]
    )
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


def weekly_genetic_algorithm(user_profiles):
    """
    Run the daily genetic algorithm for each day of the week by calling genetic_algorithm,
    aggregate results, and compute weekly nutrition/averages.
    user_profiles: list of 7 user_profile dicts (one per day)
    Returns: (daily_results, weekly_nutrition, weekly_averages)
    """
    days = 7
    daily_results = []
    weekly_nutrition = {
        k: 0.0
        for k in ["calories", "protein", "fats", "carbs", "iron", "cholesterol", "cost"]
    }
    for day in range(days):
        user_profile = user_profiles[day]
        # Delegate to the daily genetic_algorithm for each day
        best_individual, best_nutrition = genetic_algorithm(user_profile)
        daily_results.append((best_individual, best_nutrition))
        for k in weekly_nutrition:
            weekly_nutrition[k] += best_nutrition.get(k, 0.0)
    weekly_averages = {k: v / days for k, v in weekly_nutrition.items()}
    return daily_results, weekly_nutrition, weekly_averages


def main():
    while True:
        choice = main_menu()
        if choice == "1":
            user_profile = {
                "age": 21,
                "gender": "Male",
                "weight": 72,
                "height": 183,
                "activity_level": "Active",
                "goal": "gain",
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
            daily_results, weekly_nutrition, weekly_averages = weekly_genetic_algorithm(
                weekly_profiles
            )
            shopping_list = generate_weekly_shopping_list(daily_results)
            result_str = format_weekly_plan(
                daily_results, weekly_nutrition, weekly_averages, shopping_list
            )
            print(result_str)
            input("\nPress Enter to return to the main menu...")
            continue
        elif choice == "4":
            user_profile = {
                "age": 21,
                "gender": "Male",
                "weight": 72,
                "height": 183,
                "activity_level": "Active",
                "goal": "gain",
            }
            print("\nUsing default example profile for weekly planner:")
            for k, v in user_profile.items():
                print(f"  {k}: {v}")
            weekly_profiles = get_weekly_user_profiles(user_profile)
            print("\nRunning genetic algorithm to find optimal weekly diet plan...")
            daily_results, weekly_nutrition, weekly_averages = weekly_genetic_algorithm(
                weekly_profiles
            )
            shopping_list = generate_weekly_shopping_list(daily_results)
            result_str = format_weekly_plan(
                daily_results, weekly_nutrition, weekly_averages, shopping_list
            )
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
            result_str = format_meal_plan(
                best_daily_individual, best_daily_nutrition, user_profile
            )
            print(result_str)
        else:
            print("No suitable daily meal plan found.")
        input("\nPress Enter to return to the main menu...")


if __name__ == "__main__":
    main()
