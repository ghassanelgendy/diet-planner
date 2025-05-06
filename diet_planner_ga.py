# Diet Planner Genetic Algorithm
# Features:
# - Real-valued encoding
# - Tournament Selection with Elitism
# - Simulated Binary Crossover (SBC)
# - Gaussian Mutation
# - Dynamic penalty weights
# - Meal planning (breakfast, lunch, dinner)

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from collections import defaultdict

# -------------------------------------------------------------------------
# DATA PREPROCESSING
# -------------------------------------------------------------------------

def load_and_preprocess_food_data(csv_path):
    """
    Load and preprocess the food database from CSV
    Fix column names, add necessary attributes
    """
    # Load the dataset
    df = pd.read_csv(csv_path)
    
    # Fix duplicate calories column (assuming first is kcal, second is kJ)
    if 'calories' in df.columns and df.columns.duplicated().any():
        df.columns = ['fdc_id', 'description', 'protein', 'fat', 'carbs', 'calories_kcal', 'calories_kj', 'iron']
    # If no duplicates but 'calories' exists, rename it to 'calories_kcal' for consistency
    elif 'calories' in df.columns:
        df = df.rename(columns={'calories': 'calories_kcal'})
    
    # Add cost data (this would be replaced with real cost data)
    if 'cost' not in df.columns:
        # Generate random but reasonable costs based on food type
        df['cost'] = np.random.uniform(0.005, 0.02, size=len(df))  # $0.005 to $0.02 per gram
        # Make protein foods more expensive
        df.loc[df['protein'] > 15, 'cost'] *= 2.0
        # Make vegetables cheaper
        df.loc[(df['carbs'] > 5) & (df['fat'] < 1), 'cost'] *= 0.7
    
    # Classify foods by meal suitability
    df = classify_foods_by_meal(df)
    
    # Add food group classification
    df = identify_food_groups(df)
    
    return df

def classify_foods_by_meal(food_df):
    """
    Add meal suitability flags to each food item
    """
    # Define meal keywords for classification
    breakfast_keywords = ['cereal', 'milk', 'yogurt', 'egg', 'bread', 'toast', 'juice', 'fruit']
    lunch_keywords = ['sandwich', 'salad', 'soup', 'wrap', 'pasta', 'beans', 'hummus']
    dinner_keywords = ['meat', 'fish', 'poultry', 'steak', 'chicken', 'turkey', 'vegetable', 'potato', 'rice']
    
    # Initialize meal suitability flags
    food_df['breakfast_suitable'] = False
    food_df['lunch_suitable'] = False
    food_df['dinner_suitable'] = False
    
    # Classify each food
    for index, food in food_df.iterrows():
        desc_lower = food['description'].lower()
        
        # Check breakfast suitability
        if any(keyword in desc_lower for keyword in breakfast_keywords):
            food_df.at[index, 'breakfast_suitable'] = True
        
        # Check lunch suitability
        if any(keyword in desc_lower for keyword in lunch_keywords):
            food_df.at[index, 'lunch_suitable'] = True
        
        # Check dinner suitability
        if any(keyword in desc_lower for keyword in dinner_keywords):
            food_df.at[index, 'dinner_suitable'] = True
        
        # Default categorization for uncategorized items
        if not (food_df.at[index, 'breakfast_suitable'] or 
                food_df.at[index, 'lunch_suitable'] or 
                food_df.at[index, 'dinner_suitable']):
            # Set defaults based on food composition
            if food['protein'] > 15:  # High protein foods for dinner
                food_df.at[index, 'dinner_suitable'] = True
            elif food['carbs'] > 15 and food['calories_kcal'] < 300:  # Lighter carb foods for breakfast
                food_df.at[index, 'breakfast_suitable'] = True
            else:
                # Default to lunch suitable if nothing else fits
                food_df.at[index, 'lunch_suitable'] = True
    
    return food_df

def identify_food_groups(food_df):
    """
    Add food group classification to dataset
    """
    # Define food group patterns
    food_groups = {
        'protein_foods': ['beef', 'chicken', 'turkey', 'fish', 'pork', 'egg', 'sausage'],
        'dairy': ['milk', 'cheese', 'yogurt'],
        'grains': ['bread', 'rice', 'pasta', 'cereal'],
        'vegetables': ['vegetable', 'broccoli', 'carrot', 'tomato', 'kale', 'lettuce', 'onion'],
        'fruits': ['fruit', 'apple', 'orange', 'pear', 'strawberry', 'melon', 'fig', 'peach'],
        'fats': ['oil', 'seed', 'nut', 'almond', 'butter']
    }
    
    # Initialize food group columns
    for group in food_groups.keys():
        food_df[group] = False
    
    # Classify each food
    for i, food in food_df.iterrows():
        desc_lower = food['description'].lower()
        for group, keywords in food_groups.items():
            if any(keyword in desc_lower for keyword in keywords):
                food_df.at[i, group] = True
    
    return food_df

# -------------------------------------------------------------------------
# USER INPUT AND CONSTRAINT CALCULATION
# -------------------------------------------------------------------------

def get_user_profile():
    """
    Get user inputs for personalization
    In a real application, this would be a GUI form
    """
    # For demonstration, return a default user profile
    return {
        'age': 30,
        'gender': 'Male',
        'weight': 75,  # kg
        'height': 180,  # cm
        'activity_level': 'Moderate',
        'goal': 'maintain',
        'diet_type': 'Omnivore',
        'allergies': [],
        'disliked_foods': ['egg', 'fish'],
        'preferred_foods': ['chicken', 'rice']
    }

def calculate_constraints(user_profile):
    """
    Calculate nutritional constraints based on user profile
    """
    constraints = {}
    
    # Calculate Basal Metabolic Rate (BMR)
    if user_profile['gender'] == 'Male':
        bmr = 10 * user_profile['weight'] + 6.25 * user_profile['height'] - 5 * user_profile['age'] + 5
    else:
        bmr = 10 * user_profile['weight'] + 6.25 * user_profile['height'] - 5 * user_profile['age'] - 161
    
    # Apply activity multiplier
    activity_multipliers = {
        'Sedentary': 1.2,
        'Light': 1.375,
        'Moderate': 1.55,
        'Active': 1.725,
        'Very Active': 1.9
    }
    maintenance_calories = bmr * activity_multipliers[user_profile['activity_level']]
    
    # Adjust for goal
    goal_adjustments = {
        'maintain': 0,
        'lose': -500,   # 500 calorie deficit
        'gain': 500     # 500 calorie surplus
    }
    target_calories = maintenance_calories + goal_adjustments[user_profile['goal']]
    
    # Set caloric constraints (±5% flexibility)
    constraints['C_min'] = target_calories * 0.95
    constraints['C_max'] = target_calories * 1.05
    
    # Set protein constraints based on weight and activity
    if user_profile['activity_level'] in ['Active', 'Very Active']:
        constraints['P_min'] = user_profile['weight'] * 1.6  # 1.6g per kg
        constraints['P_max'] = user_profile['weight'] * 2.2  # 2.2g per kg
    else:
        constraints['P_min'] = user_profile['weight'] * 1.2  # 1.2g per kg
        constraints['P_max'] = user_profile['weight'] * 1.8  # 1.8g per kg
    
    # Set fat constraints (20-35% of calories)
    constraints['F_min'] = (target_calories * 0.20) / 9  # 9 calories per gram of fat
    constraints['F_max'] = (target_calories * 0.35) / 9
    
    # Set carb constraints (remaining calories, typically 45-65%)
    constraints['CHO_min'] = (target_calories * 0.45) / 4  # 4 calories per gram of carbs
    constraints['CHO_max'] = (target_calories * 0.65) / 4
    
    # Set micronutrient constraints
    constraints['Fe_min'] = 8 if user_profile['gender'] == 'Male' else 18  # Iron requirements
    
    # Calculate meal distribution targets
    constraints['breakfast_ratio'] = 0.25  # 25% of calories
    constraints['lunch_ratio'] = 0.35      # 35% of calories
    constraints['dinner_ratio'] = 0.40     # 40% of calories
    
    return constraints

def apply_dietary_preferences(food_data, user_profile):
    """
    Filter food database based on user dietary preferences
    """
    filtered_data = food_data.copy()
    
    # Apply diet type filter
    if user_profile['diet_type'] == 'Vegetarian':
        filtered_data = filtered_data[~filtered_data['description'].str.contains(
            'beef|pork|chicken|turkey|lamb|fish|seafood', case=False)]
    elif user_profile['diet_type'] == 'Vegan':
        filtered_data = filtered_data[~filtered_data['description'].str.contains(
            'beef|pork|chicken|turkey|lamb|fish|seafood|milk|cheese|egg|yogurt|honey', case=False)]
    
    # Apply allergy exclusions
    for allergy in user_profile['allergies']:
        filtered_data = filtered_data[~filtered_data['description'].str.contains(allergy, case=False)]
    
    return filtered_data

# -------------------------------------------------------------------------
# GENETIC ALGORITHM COMPONENTS
# -------------------------------------------------------------------------

def initialize_population(food_data, population_size=50):
    """
    Generate initial population with real-valued encoding
    Each chromosome represents quantities of each food item in grams
    """
    population = []
    n_foods = len(food_data)
    
    for _ in range(population_size):
        # Start with all zeros
        chromosome = np.zeros(n_foods)
        
        # Randomly select a few breakfast foods (2-4 items)
        breakfast_indices = food_data[food_data['breakfast_suitable']].index.tolist()
        if breakfast_indices:
            selected_indices = random.sample(breakfast_indices, min(random.randint(2, 4), len(breakfast_indices)))
            for idx in selected_indices:
                chromosome[idx] = random.uniform(20, 150)  # 20-150g portions
        
        # Randomly select a few lunch foods (3-5 items)
        lunch_indices = food_data[food_data['lunch_suitable']].index.tolist()
        if lunch_indices:
            selected_indices = random.sample(lunch_indices, min(random.randint(3, 5), len(lunch_indices)))
            for idx in selected_indices:
                chromosome[idx] = random.uniform(30, 200)  # 30-200g portions
        
        # Randomly select a few dinner foods (3-5 items)
        dinner_indices = food_data[food_data['dinner_suitable']].index.tolist()
        if dinner_indices:
            selected_indices = random.sample(dinner_indices, min(random.randint(3, 5), len(dinner_indices)))
            for idx in selected_indices:
                chromosome[idx] = random.uniform(30, 250)  # 30-250g portions
                
        population.append(chromosome)
        
    return population

def calculate_fitness(chromosome, food_data, constraints, generation=0):
    """
    Calculate fitness with dynamic penalty weights
    Lower fitness is better (minimization problem)
    """
    # Extract necessary data from food dataframe
    costs = food_data['cost'].values
    calories = food_data['calories_kcal'].values
    proteins = food_data['protein'].values
    fats = food_data['fat'].values
    carbs = food_data['carbs'].values
    iron = food_data['iron'].values
    breakfast_mask = food_data['breakfast_suitable'].values
    lunch_mask = food_data['lunch_suitable'].values
    dinner_mask = food_data['dinner_suitable'].values
    
    # Calculate total cost (objective function)
    total_cost = np.sum(chromosome * costs)
    
    # Calculate nutritional totals
    total_calories = np.sum(chromosome * calories)
    total_protein = np.sum(chromosome * proteins)
    total_fat = np.sum(chromosome * fats)
    total_carbs = np.sum(chromosome * carbs)
    total_iron = np.sum(chromosome * iron)
    
    # Calculate meal-specific nutrition
    breakfast_calories = np.sum(chromosome * calories * breakfast_mask)
    lunch_calories = np.sum(chromosome * calories * lunch_mask)
    dinner_calories = np.sum(chromosome * calories * dinner_mask)

    # New: Check if chromosome has any food items
    total_food_items = np.sum(chromosome > 0)
    if total_food_items < 8:  # Require at least 8 food items for diversity
        return 1000000 + (8 - total_food_items) * 10000  # Large penalty for too few items
    
    # ENHANCED: Higher dynamic penalty scale that increases more rapidly with generations
    # Start with higher base penalty and increase more aggressively
    penalty_scale = 5.0 + (generation / 20.0)
    
    # Initialize penalty
    penalty = 0
    
    # ENHANCED: Caloric constraints penalties with much higher weights
    if total_calories < constraints['C_min']:
        deficit_ratio = (constraints['C_min'] - total_calories) / constraints['C_min']
        penalty += deficit_ratio ** 2 * 1000 * penalty_scale  # 10x higher weight
    if total_calories > constraints['C_max']:
        excess_ratio = (total_calories - constraints['C_max']) / constraints['C_max']
        penalty += excess_ratio ** 2 * 800 * penalty_scale
    
    # ENHANCED: Protein constraints penalties with higher weights
    if total_protein < constraints['P_min']:
        deficit_ratio = (constraints['P_min'] - total_protein) / constraints['P_min']
        penalty += deficit_ratio ** 2 * 500 * penalty_scale
    if total_protein > constraints['P_max']:
        excess_ratio = (total_protein - constraints['P_max']) / constraints['P_max']
        penalty += excess_ratio ** 2 * 400 * penalty_scale
    
    # ENHANCED: Fat constraints penalties
    if total_fat < constraints['F_min']:
        deficit_ratio = (constraints['F_min'] - total_fat) / constraints['F_min']
        penalty += deficit_ratio ** 2 * 300 * penalty_scale
    if total_fat > constraints['F_max']:
        excess_ratio = (total_fat - constraints['F_max']) / constraints['F_max']
        penalty += excess_ratio ** 2 * 250 * penalty_scale
    
    # ENHANCED: Carb constraints penalties
    if total_carbs < constraints['CHO_min']:
        deficit_ratio = (constraints['CHO_min'] - total_carbs) / constraints['CHO_min']
        penalty += deficit_ratio ** 2 * 300 * penalty_scale
    if total_carbs > constraints['CHO_max']:
        excess_ratio = (total_carbs - constraints['CHO_max']) / constraints['CHO_max']
        penalty += excess_ratio ** 2 * 250 * penalty_scale
    
    # ENHANCED: Micronutrient constraints (iron) with higher weight
    if total_iron < constraints['Fe_min']:
        deficit_ratio = (constraints['Fe_min'] - total_iron) / constraints['Fe_min']
        penalty += deficit_ratio ** 2 * 200 * penalty_scale

    # NEW: Hard constraint - really penalize solutions with very low nutrition
    if total_calories < constraints['C_min'] * 0.5 or total_protein < constraints['P_min'] * 0.5:
        penalty += 100000  # Very large penalty for significantly under-nourished solutions
    
    # ENHANCED: Meal distribution penalties with stricter requirements
    # We now require at least some food in each meal category
    if breakfast_calories == 0:
        penalty += 5000 * penalty_scale  # Heavy penalty for skipping a meal
    if lunch_calories == 0:
        penalty += 5000 * penalty_scale
    if dinner_calories == 0:
        penalty += 5000 * penalty_scale
    
    # Required meal distribution if meals have calories
    total_daily_calories = constraints['C_min']  # Use target calories for calculation to avoid zero division
    
    if total_calories > 0:
        target_breakfast_cals = total_daily_calories * constraints['breakfast_ratio']
        target_lunch_cals = total_daily_calories * constraints['lunch_ratio']
        target_dinner_cals = total_daily_calories * constraints['dinner_ratio']
        
        # Breakfast distribution penalty with narrower allowed range
        if breakfast_calories < target_breakfast_cals * 0.8 or breakfast_calories > target_breakfast_cals * 1.2:
            penalty += (abs(breakfast_calories - target_breakfast_cals) / target_breakfast_cals) ** 2 * 200 * penalty_scale
        
        # Lunch distribution penalty
        if lunch_calories < target_lunch_cals * 0.8 or lunch_calories > target_lunch_cals * 1.2:
            penalty += (abs(lunch_calories - target_lunch_cals) / target_lunch_cals) ** 2 * 200 * penalty_scale
        
        # Dinner distribution penalty
        if dinner_calories < target_dinner_cals * 0.8 or dinner_calories > target_dinner_cals * 1.2:
            penalty += (abs(dinner_calories - target_dinner_cals) / target_dinner_cals) ** 2 * 200 * penalty_scale
    
    # ENHANCED: Food diversity penalties with stronger enforcement
    # This encourages including diverse food groups in the diet
    food_groups = ['protein_foods', 'dairy', 'grains', 'vegetables', 'fruits', 'fats']
    missing_groups = 0
    for group in food_groups:
        group_mask = food_data[group].values
        group_items = np.sum(chromosome * group_mask > 0)
        
        if group_items == 0:
            missing_groups += 1
            penalty += 300 * penalty_scale  # Higher penalty for missing an entire food group
    
    # Extra penalty if more than half of food groups are missing
    if missing_groups > 2:
        penalty += 1000 * missing_groups * penalty_scale
    
    # NEW: Reward for balanced solutions that meet all constraints
    # This helps guide solutions toward feasible regions of the search space
    if (total_calories >= constraints['C_min'] and total_calories <= constraints['C_max'] and
        total_protein >= constraints['P_min'] and total_protein <= constraints['P_max'] and
        total_fat >= constraints['F_min'] and total_fat <= constraints['F_max'] and
        total_carbs >= constraints['CHO_min'] and total_carbs <= constraints['CHO_max'] and
        total_iron >= constraints['Fe_min'] and
        missing_groups == 0):
        # Solution meets all constraints - provide a discount to the cost
        # This makes feasible solutions more attractive than just minimizing cost
        return total_cost * 0.8  # 20% discount on cost for feasible solutions
    
    # Return total fitness (cost + penalty)
    # Cost is now a smaller component of the overall fitness,
    # ensuring nutritional constraints are prioritized
    return total_cost + penalty

def tournament_selection(population, fitnesses, tournament_size=3):
    """
    Tournament selection - randomly select k individuals and pick the best
    """
    indices = random.sample(range(len(population)), tournament_size)
    best_idx = min(indices, key=lambda i: fitnesses[i])
    return population[best_idx].copy()

def simulated_binary_crossover(parent1, parent2, eta=2.0):
    """
    Simulated Binary Crossover (SBC)
    Produces children that are near their parents with controlled distribution
    """
    # Create copies of parents
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    if random.random() > 0.8:  # 80% crossover rate
        return child1, child2
    
    # Apply SBX to each gene
    for i in range(len(parent1)):
        # Skip if parents are identical
        if abs(parent1[i] - parent2[i]) < 1e-9:
            continue
        
        # Ensure parent1 < parent2
        if parent1[i] > parent2[i]:
            parent1[i], parent2[i] = parent2[i], parent1[i]
        
        # Calculate beta
        u = random.random()
        
        if u <= 0.5:
            beta = (2 * u) ** (1.0 / (eta + 1))
        else:
            beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
        
        # Create children
        child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
        child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
        
        # Ensure non-negative values
        child1[i] = max(0, child1[i])
        child2[i] = max(0, child2[i])
    
    return child1, child2

def gaussian_mutation(chromosome, mutation_rate=0.1, mutation_scale=0.2):
    """
    Gaussian Mutation - add normally distributed values to genes
    """
    # Create a copy of the chromosome
    mutated = chromosome.copy()
    
    for i in range(len(mutated)):
        # Apply mutation with probability mutation_rate
        if random.random() < mutation_rate:
            if mutated[i] > 0:  # Only mutate non-zero quantities
                # Add Gaussian noise
                change = np.random.normal(0, mutation_scale * mutated[i])
                mutated[i] = max(0, mutated[i] + change)  # Ensure non-negative
            else:
                # Small chance to add a food item where there was none
                if random.random() < 0.1:
                    mutated[i] = random.uniform(10, 100)  # Add small quantity
    
    return mutated

def genetic_algorithm(food_data, constraints, population_size=100, generations=200):
    """
    Main genetic algorithm loop
    """
    # Initialize population
    population = initialize_population(food_data, population_size)
    
    # Track the best solution and its fitness
    best_fitness_history = []
    avg_fitness_history = []
    best_chromosome = None
    best_fitness = float('inf')
    
    # Main GA loop
    for generation in range(generations):
        # Calculate fitness for each chromosome
        fitnesses = [calculate_fitness(chrom, food_data, constraints, generation) for chrom in population]
        
        # Track best solution
        gen_best_idx = np.argmin(fitnesses)
        gen_best_fitness = fitnesses[gen_best_idx]
        
        if gen_best_fitness < best_fitness:
            best_fitness = gen_best_fitness
            best_chromosome = population[gen_best_idx].copy()
        
        # Store history for plotting
        best_fitness_history.append(gen_best_fitness)
        avg_fitness_history.append(np.mean(fitnesses))
        
        # Create new population with elitism
        new_population = []
        
        # Elitism: Keep the best 2 individuals
        elites_idx = np.argsort(fitnesses)[:2]
        for idx in elites_idx:
            new_population.append(population[idx].copy())
        
        # Generate rest of new population
        while len(new_population) < population_size:
            # Tournament selection
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            
            # SBC Crossover
            child1, child2 = simulated_binary_crossover(parent1, parent2)
            
            # Mutation
            child1 = gaussian_mutation(child1)
            child2 = gaussian_mutation(child2)
            
            # Add to new population
            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)
        
        # Replace old population
        population = new_population
        
        # Print progress every 10 generations
        if generation % 10 == 0:
            print(f"Generation {generation}: Best Fitness = {gen_best_fitness:.2f}, Avg Fitness = {np.mean(fitnesses):.2f}")
    
    # Print final result
    print(f"\nFinal Best Fitness: {best_fitness:.2f}")
    
    # Plot fitness history
    plt.figure(figsize=(10, 5))
    plt.plot(best_fitness_history, label='Best Fitness')
    plt.plot(avg_fitness_history, label='Average Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness (lower is better)')
    plt.title('Fitness Evolution')
    plt.legend()
    plt.grid(True)
    
    return best_chromosome, best_fitness_history

# -------------------------------------------------------------------------
# OUTPUT GENERATION AND VISUALIZATION
# -------------------------------------------------------------------------

def generate_meal_plan(best_solution, food_data, constraints):
    """
    Format the GA solution into a readable meal plan
    """
    # Create lists for meal items
    breakfast_items = []
    lunch_items = []
    dinner_items = []
    
    # Calculate nutritional totals
    total_cost = 0
    total_calories = 0
    total_protein = 0
    total_fat = 0
    total_carbs = 0
    total_iron = 0
    
    breakfast_calories = 0
    lunch_calories = 0
    dinner_calories = 0
    
    # Process each food item
    for i, amount in enumerate(best_solution):
        if amount < 5:  # Ignore trace amounts
            continue
            
        food = food_data.iloc[i]
        
        # Calculate nutritional values
        food_calories = food['calories_kcal'] * amount
        food_protein = food['protein'] * amount
        food_fat = food['fat'] * amount
        food_carbs = food['carbs'] * amount
        food_iron = food['iron'] * amount
        food_cost = food['cost'] * amount
        
        # Add to overall nutritional totals
        total_calories += food_calories
        total_protein += food_protein
        total_fat += food_fat
        total_carbs += food_carbs
        total_iron += food_iron
        total_cost += food_cost
        
        # Create food entry
        food_entry = {
            'name': food['description'],
            'amount': round(amount, 1),
            'calories': round(food_calories, 1),
            'protein': round(food_protein, 1),
            'fat': round(food_fat, 1),
            'carbs': round(food_carbs, 1),
            'cost': round(food_cost, 2)
        }
        
        # Assign to appropriate meal based on meal suitability
        if food['breakfast_suitable'] and sum(item['calories'] for item in breakfast_items) < (constraints['C_max'] * constraints['breakfast_ratio']):
            breakfast_items.append(food_entry)
            breakfast_calories += food_calories
        elif food['lunch_suitable'] and sum(item['calories'] for item in lunch_items) < (constraints['C_max'] * constraints['lunch_ratio']):
            lunch_items.append(food_entry)
            lunch_calories += food_calories
        elif food['dinner_suitable']:
            dinner_items.append(food_entry)
            dinner_calories += food_calories
        else:
            # If not explicitly suitable for any meal, assign based on current distribution
            if breakfast_calories < total_calories * constraints['breakfast_ratio']:
                breakfast_items.append(food_entry)
                breakfast_calories += food_calories
            elif lunch_calories < total_calories * constraints['lunch_ratio']:
                lunch_items.append(food_entry)
                lunch_calories += food_calories
            else:
                dinner_items.append(food_entry)
                dinner_calories += food_calories
    
    # Create final meal plan
    meal_plan = {
        'breakfast': {
            'items': breakfast_items,
            'total_calories': round(breakfast_calories, 1),
            'percent_of_daily': round((breakfast_calories / total_calories) * 100 if total_calories > 0 else 0, 1)
        },
        'lunch': {
            'items': lunch_items,
            'total_calories': round(lunch_calories, 1),
            'percent_of_daily': round((lunch_calories / total_calories) * 100 if total_calories > 0 else 0, 1)
        },
        'dinner': {
            'items': dinner_items,
            'total_calories': round(dinner_calories, 1),
            'percent_of_daily': round((dinner_calories / total_calories) * 100 if total_calories > 0 else 0, 1)
        },
        'daily_totals': {
            'calories': round(total_calories, 1),
            'protein': round(total_protein, 1),
            'fat': round(total_fat, 1),
            'carbs': round(total_carbs, 1),
            'iron': round(total_iron, 2),
            'cost': round(total_cost, 2)
        }
    }
    
    return meal_plan

def print_meal_plan(meal_plan, constraints):
    """
    Display the meal plan in a user-friendly format
    """
    print("\n" + "=" * 80)
    print("OPTIMAL DIET PLAN")
    print("=" * 80)
    
    # Print breakfast
    print("\nBREAKFAST ({:.1f} calories, {:.1f}% of daily total)".format(
        meal_plan['breakfast']['total_calories'], 
        meal_plan['breakfast']['percent_of_daily']))
    print("-" * 80)
    
    if meal_plan['breakfast']['items']:
        for item in meal_plan['breakfast']['items']:
            print(f"{item['amount']:6.1f}g {item['name']:32s} {item['calories']:6.1f} cal " + 
                 f"{item['protein']:5.1f}g protein {item['fat']:5.1f}g fat {item['carbs']:5.1f}g carbs ${item['cost']:.2f}")
    else:
        print("No breakfast items selected.")
    
    # Print lunch
    print("\nLUNCH ({:.1f} calories, {:.1f}% of daily total)".format(
        meal_plan['lunch']['total_calories'], 
        meal_plan['lunch']['percent_of_daily']))
    print("-" * 80)
    
    if meal_plan['lunch']['items']:
        for item in meal_plan['lunch']['items']:
            print(f"{item['amount']:6.1f}g {item['name']:32s} {item['calories']:6.1f} cal " + 
                 f"{item['protein']:5.1f}g protein {item['fat']:5.1f}g fat {item['carbs']:5.1f}g carbs ${item['cost']:.2f}")
    else:
        print("No lunch items selected.")
    
    # Print dinner
    print("\nDINNER ({:.1f} calories, {:.1f}% of daily total)".format(
        meal_plan['dinner']['total_calories'], 
        meal_plan['dinner']['percent_of_daily']))
    print("-" * 80)
    
    if meal_plan['dinner']['items']:
        for item in meal_plan['dinner']['items']:
            print(f"{item['amount']:6.1f}g {item['name']:32s} {item['calories']:6.1f} cal " + 
                 f"{item['protein']:5.1f}g protein {item['fat']:5.1f}g fat {item['carbs']:5.1f}g carbs ${item['cost']:.2f}")
    else:
        print("No dinner items selected.")
    
    # Print daily totals
    print("\n" + "=" * 80)
    print("DAILY NUTRITIONAL SUMMARY")
    print("=" * 80)
    print(f"Total Calories: {meal_plan['daily_totals']['calories']:.1f} cal (Target: {constraints['C_min']:.1f} - {constraints['C_max']:.1f})")
    print(f"Total Protein:  {meal_plan['daily_totals']['protein']:.1f}g (Target: {constraints['P_min']:.1f} - {constraints['P_max']:.1f})")
    print(f"Total Fat:      {meal_plan['daily_totals']['fat']:.1f}g (Target: {constraints['F_min']:.1f} - {constraints['F_max']:.1f})")
    print(f"Total Carbs:    {meal_plan['daily_totals']['carbs']:.1f}g (Target: {constraints['CHO_min']:.1f} - {constraints['CHO_max']:.1f})")
    print(f"Total Iron:     {meal_plan['daily_totals']['iron']:.2f}mg (Target: ≥ {constraints['Fe_min']:.1f})")
    print(f"Total Cost:     ${meal_plan['daily_totals']['cost']:.2f}")
    print("=" * 80)

def visualize_nutritional_distribution(meal_plan, constraints):
    """
    Create visual representations of the nutritional breakdown
    """
    # Setup figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Meal caloric distribution pie chart
    meal_cals = [
        meal_plan['breakfast']['total_calories'],
        meal_plan['lunch']['total_calories'],
        meal_plan['dinner']['total_calories']
    ]
    
    ax1.pie(
        meal_cals,
        labels=['Breakfast', 'Lunch', 'Dinner'],
        autopct='%1.1f%%',
        startangle=90,
        colors=['#FF9999', '#66B2FF', '#99FF99']
    )
    ax1.set_title('Caloric Distribution by Meal')
    
    # 2. Macronutrient distribution compared to targets
    macros = [
        meal_plan['daily_totals']['protein'], 
        meal_plan['daily_totals']['fat'], 
        meal_plan['daily_totals']['carbs']
    ]
    
    macro_targets_min = [constraints['P_min'], constraints['F_min'], constraints['CHO_min']]
    macro_targets_max = [constraints['P_max'], constraints['F_max'], constraints['CHO_max']]
    
    x = np.arange(3)
    width = 0.25
    
    ax2.bar(x - width, macros, width, label='Actual', color='#66B2FF')
    ax2.bar(x, macro_targets_min, width, label='Min Target', color='#FF9999', alpha=0.7)
    ax2.bar(x + width, macro_targets_max, width, label='Max Target', color='#99FF99', alpha=0.7)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Protein (g)', 'Fat (g)', 'Carbs (g)'])
    ax2.set_title('Macronutrients vs. Targets')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------------------------

def main():
    # 1. Load and preprocess food data
    print("Loading and preprocessing food data...")
    food_data = load_and_preprocess_food_data("datasets/cleaned_food_database.csv")
    
    # 2. Get user profile
    print("\nGetting user profile...")
    user_profile = get_user_profile()
    
    # 3. Calculate constraints
    print("\nCalculating nutritional constraints...")
    constraints = calculate_constraints(user_profile)
    
    # 4. Apply dietary preferences
    print("\nApplying dietary preferences...")
    filtered_food_data = apply_dietary_preferences(food_data, user_profile)
    
    # 5. Run genetic algorithm
    print("\nRunning genetic algorithm optimization...")
    best_solution, fitness_history = genetic_algorithm(
        filtered_food_data, 
        constraints,
        population_size=80,
        generations=150
    )
    
    # 6. Generate meal plan
    print("\nGenerating optimized meal plan...")
    meal_plan = generate_meal_plan(best_solution, filtered_food_data, constraints)
    
    # 7. Print and visualize results
    print("\nFinal diet plan:")
    print_meal_plan(meal_plan, constraints)
    visualize_nutritional_distribution(meal_plan, constraints)

if __name__ == "__main__":
    main()