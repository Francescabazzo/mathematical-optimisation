"""
numerical_experiments.py 

This script evaluates the performance (mean matching score) of multiple rule-based heuristic variations
and linear programming variations, where the size of the sets are the same (|S|=|M|). 
Each algorithm was tested on different sizes of the sets and, for each time, the computation was iterated
100 times, creating each time a different dataset. 

Other configurations were also tested, but are commented out: 
- |M| = 0.9*|S| (more mentees than mentors)
- |M| = 1.1*|S| (more mentors than mentees)
"""

import pandas as pd 
import time 
from dataset_generator import generate_dataset
from test import run_linear_programming, run_linear_programming_alternative, run_heuristic, compute_mean_score 


# =============================================================
# Matching criteria definitions for rule-based heuristics (RB1–RB6)
# =============================================================

#   RB1 
matching_criteria_1 = [
    ("gender", "type", "programme", "subject"),     #   Iteration 1 
    ("type", "programme", "subject"),               #   Iteration 2 (Filtering criteria "no_pref" in gender)
    ("gender", "type", "programme"),                #   Iteration 3
    ("type", "programme"),                          #   Iteration 4 (Filtering criteria "no_pref" in gender)
    ("gender", "type"),                             #   Iteration 5
    ("type",),                                      #   Iteration 6 (Filtering criteria "no_pref" in gender)
    ("gender",),                                    #   Iteration 7 
    ("type", "programme"),                          #   Iteration 8 
    ("type",),                                      #   Iteration 9
    ("programme",),                                 #   Iteration 10
    ()                                              #   Iteration 11
]
steps_with_filtering_1 = {2, 4, 6}

#   RB2 
matching_criteria_2 = [
    ("gender", "type", "programme"),    #   Iteration 1 
    ("type", "programme"),              #   Iteration 2 (Filtering criteria "no_pref" in gender)
    ("gender", "type"),                 #   Iteration 3
    ("type",),                          #   Iteration 4 (Filtering criteria "no_pref" in gender)
    ("gender",),                        #   Iteration 5
    ("type", "programme"),              #   Iteration 6 (Filtering criteria "no_pref" in gender)
    ("type",),                          #   Iteration 7 
    ("programme",),                     #   Iteration 8 
    ()                                  #   Iteration 9   
]
steps_with_filtering_2 = {2, 4}

#   RB3 
matching_criteria_3 = [
    ("gender", "type", "programme"),    #   Iteration 1 
    ("type", "programme"),              #   Iteration 2 (Filtering criteria "no_pref" in gender)
    ("gender", "type"),                 #   Iteration 3
    ("type",),                          #   Iteration 4 (Filtering criteria "no_pref" in gender)
    ("gender",),                        #   Iteration 5
    ()                                  #   Iteration 6   
]
steps_with_filtering_3 = {2, 4}

#   RB4 
matching_criteria_4 = [
    ("gender", "type", "programme"),    #   Iteration 1 
    ("type", "programme"),              #   Iteration 2 (Filtering criteria "no_pref" in gender)
    ("gender", "type"),                 #   Iteration 3
    ()                                  #   Iteration 4   
]
steps_with_filtering_4 = {2} 

#   RB5
matching_criteria_5 = [
    ("gender", "type", "programme"),    #   Iteration 1 
    ("gender",),                        #   Iteration 2
    ()                                  #   Iteration 3   
]

#   RB6 
matching_criteria_6 = [
    ("gender", "type"),                 #   Iteration 1
    ()                                  #   Iteration 2
]



# =============================================================
# Weight definitions for LP variants (LP1–LP6)
# =============================================================

#   LP1: gender priority
weights_LP1 = {"gender": 0.5, "type": 0.25, "programme": 0.15, "subject": 0.1}

#   LP2: equal priority 
weights_LP2 = {"gender": 0.25, "type": 0.25, "programme": 0.25, "subject": 0.25}

#   LP3: type and gender only 
weights_LP3 = {"gender": 0.5, "type": 0.5, "programme": 0, "subject": 0}

#   LP4: gender only 
weights_LP4 = {"gender": 1, "type": 0, "programme": 0, "subject": 0} 

#   LP5: type only 
weights_LP5 = {"gender": 0, "type": 1, "programme": 0, "subject": 0}

#   LP6: programme only 
weights_LP6 = {"gender": 0, "type": 0, "programme": 1, "subject": 0}


# =============================================================
# Experiment parameters and containers
# =============================================================

n_replications = 100

results_small = []
results_medium = []
results_large = []


# =============================================================
# Function run_iterations 
# =============================================================

def run_iterations(size_mentees, size_mentors, n_replications=1000):
    """
    Runs the matching algorithms multiple times on randomly generated datasets.
    Computes the mean matching score for each run and stores it by dataset size.

    Parameters
    ----------
    size_mentees : int
        Number of mentees.
    size_mentors : int
        Number of mentors.
    n_replications : int, optional
        Number of times to repeat the experiment (default is 1000).
    """
    #   Approximate to the closest int if the input is a float
    size_mentees = int(round(size_mentees))
    size_mentors = int(round(size_mentors))

    if size_mentees <= 0 or size_mentors <= 0:
        print("The sizes of the two sets needs to be a positive integer number.")
        return 

    #   Clear previous results for the same size group 
    if size_mentees <= 50:
        results_small.clear()
    elif 50 < size_mentees < 150:
        results_medium.clear()
    else:
        results_large.clear()

    start_total = time.time()   
    last_10times = []           
    start_batch = time.time()   

    for rep in range(n_replications):
        start_iter = time.time()    
        
        #   Generate synthetic datasets
        mentees, mentors = generate_dataset(size_mentees, size_mentors, 1)

        #   Rule-based heuristics 
        score_h1, matching_h1, time_h1 = run_heuristic(mentees, mentors, matching_criteria_1, weights_LP1, steps_with_filtering_1)
        score_h2, matching_h2, time_h2 = run_heuristic(mentees, mentors, matching_criteria_2, weights_LP1, steps_with_filtering_2)
        score_h3, matching_h3, time_h3 = run_heuristic(mentees, mentors, matching_criteria_3, weights_LP1, steps_with_filtering_3)
        score_h4, matching_h4, time_h4 = run_heuristic(mentees, mentors, matching_criteria_4, weights_LP1, steps_with_filtering_4)
        score_h5, matching_h5, time_h5 = run_heuristic(mentees, mentors, matching_criteria_5, weights_LP1)
        score_h6, matching_h6, time_h6 = run_heuristic(mentees, mentors, matching_criteria_6, weights_LP1)
        
        mean_h1 = compute_mean_score(matching_h1, n_mentees=size_mentees)
        mean_h2 = compute_mean_score(matching_h2, n_mentees=size_mentees)
        mean_h3 = compute_mean_score(matching_h3, n_mentees=size_mentees)
        mean_h4 = compute_mean_score(matching_h4, n_mentees=size_mentees)
        mean_h5 = compute_mean_score(matching_h5, n_mentees=size_mentees)
        mean_h6 = compute_mean_score(matching_h6, n_mentees=size_mentees)

        #   Linear programming 
        if size_mentees > size_mentors:
            Z_lp1, matching_lp1, time_lp1 = run_linear_programming_alternative(mentees, mentors, weights_LP1)    
            Z_lp2, matching_lp2, time_lp2 = run_linear_programming_alternative(mentees, mentors, weights_LP2)
            Z_lp3, matching_lp3, time_lp3 = run_linear_programming_alternative(mentees, mentors, weights_LP3)
            Z_lp4, matching_lp4, time_lp4 = run_linear_programming_alternative(mentees, mentors, weights_LP4)
            Z_lp5, matching_lp5, time_lp5 = run_linear_programming_alternative(mentees, mentors, weights_LP5)
            Z_lp6, matching_lp6, time_lp6 = run_linear_programming_alternative(mentees, mentors, weights_LP6)
        else: 
            Z_lp1, matching_lp1, time_lp1 = run_linear_programming(mentees, mentors, weights_LP1)
            Z_lp2, matching_lp2, time_lp2 = run_linear_programming(mentees, mentors, weights_LP2)
            Z_lp3, matching_lp3, time_lp3 = run_linear_programming(mentees, mentors, weights_LP3)
            Z_lp4, matching_lp4, time_lp4 = run_linear_programming(mentees, mentors, weights_LP4)
            Z_lp5, matching_lp5, time_lp5 = run_linear_programming(mentees, mentors, weights_LP5)
            Z_lp6, matching_lp6, time_lp6 = run_linear_programming(mentees, mentors, weights_LP6)
        
        mean_lp1 = compute_mean_score(matching_lp1, n_mentees=size_mentees)
        mean_lp2 = compute_mean_score(matching_lp2, n_mentees=size_mentees)
        mean_lp3 = compute_mean_score(matching_lp3, n_mentees=size_mentees)
        mean_lp4 = compute_mean_score(matching_lp4, n_mentees=size_mentees)
        mean_lp5 = compute_mean_score(matching_lp5, n_mentees=size_mentees)
        mean_lp6 = compute_mean_score(matching_lp6, n_mentees=size_mentees)

        # Save results 
        if size_mentees <= 50: 
            results_small.append({
                "rep": rep,
                "RB1_mean": mean_h1,
                "RB2_mean": mean_h2,
                "RB3_mean": mean_h3,
                "RB4_mean": mean_h4,
                "RB5_mean": mean_h5,
                "RB6_mean": mean_h6,
                "LP1_mean": mean_lp1,
                "LP2_mean": mean_lp2,
                "LP3_mean": mean_lp3,
                "LP4_mean": mean_lp4,
                "LP5_mean": mean_lp5,
                "LP6_mean": mean_lp6
            }) 
        elif size_mentees > 50 and size_mentees < 150: 
            results_medium.append({
                "rep": rep,
                "RB1_mean": mean_h1,
                "RB2_mean": mean_h2,
                "RB3_mean": mean_h3,
                "RB4_mean": mean_h4,
                "RB5_mean": mean_h5,
                "RB6_mean": mean_h6,
                "LP1_mean": mean_lp1,
                "LP2_mean": mean_lp2,
                "LP3_mean": mean_lp3,
                "LP4_mean": mean_lp4,
                "LP5_mean": mean_lp5,
                "LP6_mean": mean_lp6
            })
        else:
            results_large.append({
                "rep": rep,
                "RB1_mean": mean_h1,
                "RB2_mean": mean_h2,
                "RB3_mean": mean_h3,
                "RB4_mean": mean_h4,
                "RB5_mean": mean_h5,
                "RB6_mean": mean_h6,
                "LP1_mean": mean_lp1,
                "LP2_mean": mean_lp2,
                "LP3_mean": mean_lp3,
                "LP4_mean": mean_lp4,
                "LP5_mean": mean_lp5,
                "LP6_mean": mean_lp6
            })
        
        end_iter = time.time()                  
        elapsed_iter = end_iter - start_iter    
        last_10times.append(elapsed_iter)       

        # Progress update every 10 iterations
        if (rep + 1) % 10 == 0:
            end_batch = time.time()
            elapsed_batch = end_batch - start_batch
            print(f"Size {size_mentees}, iteration {rep + 1}/{n_replications} - time last 10 iters: {elapsed_batch:.2f} sec")
            start_batch = time.time()

    end_total = time.time()
    total_elapsed = end_total - start_total
    print(f"Finished run_iterations for size {size_mentees} with {n_replications} reps in {total_elapsed:.2f} seconds")


# ============================================
# Run experiments for |S|=|M| only
# ============================================

#   |S|=|M|=20
run_iterations(20, 18, 100)

#   |S|=|M|=100
run_iterations(100, 100, 100)

#   |S|=|M|=250
run_iterations(250, 250, 100)

# ============================================
# Summary statistics 
# ============================================

methods = ["RB1_mean", "RB2_mean", "RB3_mean", "RB4_mean", "RB5_mean", "RB6_mean",
           "LP1_mean", "LP2_mean", "LP3_mean", "LP4_mean", "LP5_mean", "LP6_mean"]


#   SMALL dataset 

df_small = pd.DataFrame(results_small)

#   I create a dataframe where I compute the cumulative values 
summary_small = pd.DataFrame({
    "Mean": df_small[methods].mean(),
    "Median": df_small[methods].median(),
    "Std": df_small[methods].std()
})

summary_small = summary_small.reset_index().rename(columns={"index": "Method"})

print("\n==== Results for SMALL dataset (|S|=20) ====")
print(summary_small.round(3))


#   MEDIUM DATASET 

df_medium = pd.DataFrame(results_medium)

#   Dataframe where I compute the cumulative values 
summary_medium = pd.DataFrame({
    "Mean": df_medium[methods].mean(),
    "Median": df_medium[methods].median(),
    "Std": df_medium[methods].std()
})

summary_medium = summary_medium.reset_index().rename(columns={"index": "Method"})

print("\n==== Results for MEDIUM dataset (|S|=100) ====")
print(summary_medium.round(3))


#   LARGE DATASET 

df_large = pd.DataFrame(results_large)

#   Dataframe where I compute the cumulative values 
summary_large = pd.DataFrame({
    "Mean": df_large[methods].mean(),
    "Median": df_large[methods].median(),
    "Std": df_large[methods].std()
})

summary_large = summary_large.reset_index().rename(columns={"index": "Method"})

print("\n==== Results for LARGE dataset (|S|=250) ====")
print(summary_large.round(3))