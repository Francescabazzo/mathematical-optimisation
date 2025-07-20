"""
scalability.py 

This script evaluates the scalability of four matching algorithms (LP1, LP2, RBi, RBii)
across increasing dataset sizes (|S| = |M|). Each algorithm is tested on synthetic data
generated consistently for fairness in comparison.

Algorithms tested:
- LP1: Linear Programming with priority on gender
- LP2: Linear Programming with priority on gender and type
- RBi: Rule-Based iterative with multiple filtering steps
- RBii: Rule-Based simplified (2 steps only)

"""


import matplotlib.pyplot as plt
from dataset_generator import generate_dataset
from test import run_linear_programming, run_heuristic, compute_total_score, compute_mean_score
import pandas as pd


sizes = [10, 50, 100, 500, 1000, 2000]
results = []

# LP1: gender priority
weights_LP1 = {
    "gender": 0.5,
    "type": 0.25,
    "programme": 0.15,
    "subject": 0.1
}

# LP2: gender and type
weights_LP2 = {
    "gender": 0.5,
    "type": 0.5,
    "programme": 0,
    "subject": 0
}

# RBi criteria and filtering steps
matching_criteria_1 = [
    ("gender", "type", "programme"),
    ("type", "programme"),
    ("gender", "type"),
    ("type",),
    ("gender",),
    ()
]
steps_with_filtering_1 = {2, 4}

# RBii criteria 
matching_criteria_2 = [
    ("gender", "type", "programme"),    #   Iteration 1 
    ()                                  #   Iteration 2
]

# ---------- Run experiments ----------

for size in sizes:
    print(f"\n--- Testing size = {size} ---")

    mentees, mentors = generate_dataset(num_mentees=size, num_mentors=size, scenario=1)

    # LP1
    Z_lp1, match_lp1, time_lp1 = run_linear_programming(mentees, mentors, weights_LP1)
    total_lp1 = compute_total_score(match_lp1)
    mean_lp1 = compute_mean_score(match_lp1, n_mentees=size)
    print(f" -> LP1 completed for size {size}")

    # LP2
    Z_lp2, match_lp2, time_lp2 = run_linear_programming(mentees, mentors, weights_LP2)
    total_lp2 = compute_total_score(match_lp2)
    mean_lp2 = compute_mean_score(match_lp2, n_mentees=size)
    print(f" -> LP2 completed for size {size}")


    # RBi
    total_rbi, match_rbi, time_rbi = run_heuristic(
        mentees, mentors, matching_criteria_1, weights_LP1, steps_with_filtering_1
    )
    mean_rbi = compute_mean_score(match_rbi, n_mentees=size)
    print(f" -> RBi completed for size {size}")

    # RBii
    total_rbii, match_rbii, time_rbii = run_heuristic(
        mentees, mentors, matching_criteria=matching_criteria_2, weights=weights_LP1
    )
    mean_rbii = compute_mean_score(match_rbii, n_mentees=size)
    print(f" -> RBii completed for size {size}")

    # Store results
    results.append({
        "size": size,
        "LP1_time": round(time_lp1, 3),
        "LP1_total": total_lp1,
        "LP1_mean": round(mean_lp1, 3),
        "LP2_time": round(time_lp2,3 ),
        "LP2_total": total_lp2,
        "LP2_mean": round(mean_lp2, 3),
        "RBi_time": round(time_rbi, 3),
        "RBi_total": total_rbi,
        "RBi_mean": round(mean_rbi, 3),
        "RBii_time": round(time_rbii, 3),
        "RBii_total": total_rbii,
        "RBii_mean": round(mean_rbii, 3)
    })


# ---------- Save and show results ----------

df = pd.DataFrame(results)
df.to_csv("scalability_results.csv", index=False)
print("\n✅ Results saved to scalability_results.csv")
print(df)

