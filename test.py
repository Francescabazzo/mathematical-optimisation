"""
test.py file 

This file contains implementations for:
- scoring function between mentees and mentors based on preferences and attributes
- computation of total, mean and minimum scores of a matching
- two linear programming models for optimal matching
- two rule-based heuristics (one detailed with 6 steps, one simplified with 2 steps)

It also provides a simple example with small datasets to test the methods.
"""


# ============================================
# LIBRARIES AND DATA IMPORT
# ============================================

import gurobipy as gb 
import pandas as pd 
import numpy as np 
import time 
from dataset_generator import generate_dataset

mentees_se_df = pd.read_csv("data/mentees_se.csv")
mentors_se_df = pd.read_csv("data/mentors_se.csv")


# ============================================
# SCORING FUNCTIONS
# ============================================

def weights_score(mentee, mentor, weights):
    """
    Parameters
    ----------
    mentee : pd.Series
        A row from the mentee dataframe.
    mentor : pd.Series
        A row from the mentor dataframe.
    weights : dict
        Dictionary specifying the weight of each attribute: gender, type, programme, subject.

    Returns
    -------
    float
        Total compatibility score for the mentee-mentor pair.
    """
    score = 0
    # Gender: con flessibilità se no_pref
    if mentee["gender_pref"] == "no_pref" or mentee["gender_pref"] == mentor["gender"]:
        score += weights["gender"]
    # Type
    if mentee["type"] == mentor["type"]:
        score += weights["type"]
    # Programme (one-to-many)
    if "-" not in mentor["programme"]:
        if mentee["programme"] in mentor["programme"]:
            score += weights["programme"]
    # Subject (one-to-many)
    if "-" not in mentor["subject"]: 
        if mentee["subject"] in mentor["subject"]:
            score += weights["subject"]
    return score


def compute_total_score(matching, index=2): 
    """
    Compute the total score for all mentee-mentor matchings.

    Parameters
    ----------
    matching : list of tuples
        Each tuple represents a matching and includes the score.
    index : int, optional
        Index of the score in the tuple (default is 2).

    Returns
    -------
    float
        Sum of all scores.
    """
    scores = [match[index] for match in matching]
    total = sum(scores)
    return total

def compute_mean_score(matching, n_mentees, index=2):
    """
    Compute the mean score for all mentee-mentor matchings.

    Parameters
    ----------
    matching : list of tuples
        Each tuple represents a matching and includes the score.
    n_mentees: int
        Number of mentees
    index : int, optional
        Index of the score in the tuple (default is 2).

    Returns
    -------
    float
        Mean score across all matchings.
    """
    scores = [match[index] for match in matching]
    total = sum(scores)
    mean = total/n_mentees if scores else 0
    return mean

def compute_min_score(matching, index=2):
    """
    Compute the minimum score among all mentee-mentor matchings.

    Parameters
    ----------
    matching : list of tuples
        Each tuple represents a matching and includes the score.
    index : int, optional
        Index of the score in the tuple (default is 2).

    Returns
    -------
    float
        Minimum score in the matching.
    """
    scores = [match[index] for match in matching]
    minimum = min(scores) if scores else 0
    return minimum 


"""
LINEAR PROGRAMMING MATCHING FUNCTIONS

This section defines two optimization-based approaches to solve the mentor–mentee matching problem using Gurobi.

Two models are implemented:
- LP:  standard one-to-one assignment model, where each mentee is assigned to exactly one mentor,
       and each mentor can be assigned to at most one mentee.
- LP_alternative: an alternative many-to-one formulation where each mentor can be matched with up to 3 mentees.
       This model is useful in numerical experiments for cases with more mentees than mentors.

In both models:
- The objective is to maximize the total matching score across all pairs.
- Compatibility scores are computed using the `weights_score` function with a specified weight set.
- The returned matching always uses `weights_LP1` to remain consistent with evaluation metrics in the paper.

Each function returns:
- The optimal objective value Z (from the model),
- The final matching as a list of tuples (mentee_id, mentor_id, score),
- The time taken to compute the solution.
"""

#   Predefined weight dictionaries 

#   Standard weights dictionary 
weights_LP1 = {
    "gender": 0.5,
    "type": 0.25,
    "programme": 0.15,
    "subject": 0.1
}

weights_LP2 = {
    "gender": 0.5,
    "type": 0.5, 
    "programme": 0, 
    "subject": 0
} 


def run_linear_programming(mentees, mentors, weights): 
    """
    Solve the mentee-mentor assignment problem assuming one-to-one matching:
    each mentee assigned exactly to one mentor and each mentor to at most one mentee.

    Parameters
    ----------
    mentees : pd.DataFrame
        DataFrame containing mentees data.
    mentors : pd.DataFrame
        DataFrame containing mentors data.
    weights : dict
        Dictionary of weights used in the scoring function.

    Returns
    -------
    tuple
        (objective_value, matching, computation_time)
        - objective_value (float): optimal total matching score.
        - matching (list of tuples): list of (mentee_id, mentor_id, score).
        - computation_time (float): elapsed time in seconds.
    """
    start = time.time()

    m = gb.Model("Test") 
    m.setParam('OutputFlag', 0)


    S = mentees.shape[0]      #   number of mentees
    M = mentors.shape[0]      #   number of mentors

    #   Matrix of the weights 
    r_matrix = np.zeros((S,M))
    for i in range(S):
        mentee = mentees.iloc[i]
        for j in range(M):
            mentor = mentors.iloc[j]
            r_matrix[i][j] = weights_score(mentee, mentor, weights)

    #   variable
    x=m.addVars([(i,j) for i in range(S) for j in range(M)], vtype=gb.GRB.BINARY, name="x")

    #   constraints
    for i in range(S): 
        m.addConstr(gb.quicksum(x[i,j] for j in range(M)) == 1)

    for j in range(M):
        m.addConstr(gb.quicksum(x[i,j] for i in range(S)) <= 1)

    #   objective function 
    m.setObjective(
        gb.quicksum(r_matrix[i][j]*x[i,j] for i in range(S) for j in range(M)), 
        gb.GRB.MAXIMIZE
    )

    m.optimize() 

    computation_time = time.time() - start 
    
    matching = [] 
    
    if m.status == gb.GRB.OPTIMAL:
        for i in range(S):
            for j in range(M):
                if x[i,j].X > 0.5:
                    mentee = mentees.iloc[i]
                    mentor = mentors.iloc[j]
                    score = weights_score(mentee, mentor, weights_LP1)  #   Score computed with standard weights
                    matching.append((mentees.iloc[i]['id'], mentors.iloc[j]['id'], score))
        return m.objVal, matching, computation_time
    else:
        print("No optimal solution found.")


def run_linear_programming_alternative(mentees, mentors, weights): 
    """
    Alternate formulation where each mentee is assigned exactly one mentor,
    but each mentor can have up to 3 mentees.

    Parameters
    ----------
    mentees : pd.DataFrame
        DataFrame containing mentees data.
    mentors : pd.DataFrame
        DataFrame containing mentors data.
    weights : dict
        Dictionary of weights used in the scoring function.

    Returns
    -------
    tuple
        (objective_value, matching, computation_time)
        - objective_value (float): optimal total matching score.
        - matching (list of tuples): list of (mentee_id, mentor_id, score).
        - computation_time (float): elapsed time in seconds.
    """
    start = time.time()

    m = gb.Model("Test_Alternate") 
    m.setParam('OutputFlag', 0)


    S = mentees.shape[0]      #   number of mentees
    M = mentors.shape[0]      #   number of mentors

    #   Matrix of the weights 
    r_matrix = np.zeros((S,M))
    for i in range(S):
        mentee = mentees.iloc[i]
        for j in range(M):
            mentor = mentors.iloc[j]
            r_matrix[i][j] = weights_score(mentee, mentor, weights)

    #   variable
    x=m.addVars([(i,j) for i in range(S) for j in range(M)], vtype=gb.GRB.BINARY, name="x")

    #   constraints
    for i in range(S): 
        m.addConstr(gb.quicksum(x[i,j] for j in range(M)) == 1)

    for j in range(M):
        m.addConstr(gb.quicksum(x[i,j] for i in range(S)) <= 3)

    #   objective function 
    m.setObjective(
        gb.quicksum(r_matrix[i][j]*x[i,j] for i in range(S) for j in range(M)), 
        gb.GRB.MAXIMIZE
    )

    m.optimize() 

    computation_time = time.time() - start 
    
    matching = [] 
    
    if m.status == gb.GRB.OPTIMAL:
        for i in range(S):
            for j in range(M):
                if x[i,j].X > 0.5:
                    mentee = mentees.iloc[i]
                    mentor = mentors.iloc[j]
                    score = weights_score(mentee, mentor, weights_LP1)  #   score computed with standard weights
                    matching.append((mentees.iloc[i]['id'], mentors.iloc[j]['id'], score))
        return m.objVal, matching, computation_time
    else:
        print("No optimal solution found.")



"""
HEURISTIC MATCHING FUNCTIONS

This section defines the multi-step rule-based heuristic algorithm used to match mentees with mentors.
Each step applies specific grouping and optionally filters mentees with 'no_pref' gender preference.

Overview of the steps:
0. Initialize the matching (D) and copy datasets of unmatched mentees (D_S) and mentors (D_M)
1. If required, filter mentees with 'no_pref' gender preference
2. Sort mentees by current matching criteria and create group/index keys
3. Sort mentors by current matching criteria and create group/index keys
4. Match mentees and mentors by group and index
5. Remove matched mentees from D_S
6. Remove matched mentors from D_M
7. Repeat until all steps are completed or either dataset is empty
"""

#   Matching criteria groups used in the heuristic RBi
matching_criteria_1 = [
    ("gender", "type", "programme"),    #   Iteration 1
    ("type", "programme"),              #   Iteration 2 (Filtering criteria "no_preference")
    ("gender", "type"),                 #   Iteration 3
    ("type",),                          #   Iteration 4 (Filtering criteria "no_preference")
    ("gender",),                        #   Iteration 5
    ()                                  #   Iteration 6 (no criteria)    
]

#   Matching criteria groups used in the heuristic RBii 
matching_criteria_2 = [
    ("gender", "type", "programme"),    #   Iteration 1 
    ()                                  #   Iteration 2
]

#   Filtering steps used in heuristic RBi 
steps_with_filtering_1 = {2, 4}


def run_heuristic(mentees, mentors, matching_criteria, weights, steps_with_filtering=None):
    """
    Runs the heuristic matching algorithm between mentees and mentors over multiple iterations.

    Parameters
    ----------
    mentees : pd.DataFrame
        DataFrame containing mentee information.
    mentors : pd.DataFrame
        DataFrame containing mentor information.
    matching_criteria : list of tuples
        Each tuple contains the attributes to match on in that iteration.
    weights : dict 
        Parameters for the scoring function `weights_score`.
    steps_with_filtering : set, optional
        Set of iteration steps where filtering mentees with 'no_pref' gender preference is applied.

    Returns
    -------
    total_score : float
        Sum of matching scores obtained.
    matching : list of tuples
        Each tuple contains (mentee_id, mentor_id, score, iteration).
    computation_time : float
        Time taken to run the heuristic.
    """
    start = time.time()

    if steps_with_filtering is None:
        steps_with_filtering = set() 

    #   Step 0: initialisation of the D_S, D_M and D (matching)
    unmatched_mentees = mentees.copy() 
    unmatched_mentors = mentors.copy() 

    matching = []

    for iteration, group in enumerate(matching_criteria, start=1): 
        
        #   Step 7: if D_S or D_M is empty, stop 
        if unmatched_mentees.empty or unmatched_mentors.empty:
            break

        #   Step 1: eventual filtering step 
        if iteration in steps_with_filtering:
            mentees_touse = unmatched_mentees[unmatched_mentees["gender_pref"] == "no_pref"]
        else:
            mentees_touse = unmatched_mentees

        if group:

            mentees_key = mentees_touse.copy()
            mentors_key = unmatched_mentors.copy() 
            
            #   Step 2 and 3: sort, select and number mentees and mentors 

            #   Create a group variable (the column is named "group_key") representing each unique 
            #   combination of the matching criteria 
            mentees_key["group_key"] = list(zip(*(mentees_key[attr if attr != "gender" else "gender_pref"] for attr in group)))
            mentors_key["group_key"] = list(zip(*(
                mentors_key["preferred_programme"] if attr == "programme"
                else mentors_key["preferred_subject"] if attr == "subject"
                else mentors_key[attr]
                for attr in group
            )))

            #   Sort by the current matching criteria
            group_mentees = [attr if attr != "gender" else "gender_pref" for attr in group]
            mentees_key = mentees_key.sort_values(group_mentees)
            group_mentors = [
                "preferred_programme" if attr == "programme"
                else "preferred_subject" if attr == "subject"
                else attr
                for attr in group
            ]
            mentors_key = mentors_key.sort_values(list(group_mentors))

            #   Add an index variable numbering the mentees and the mentors within each group 
            mentees_key["group_index"] = mentees_key.groupby("group_key").cumcount()
            mentors_key["group_index"] = mentors_key.groupby("group_key").cumcount() 

        else: 
            mentees_key = mentees_touse.copy()
            mentors_key = unmatched_mentors.copy()

            mentees_key["group_key"] = "all"
            mentors_key["group_key"] = "all"

            mentees_key = mentees_key.sort_values("id")
            mentors_key = mentors_key.sort_values("id")

            mentees_key["group_index"] = range(len(mentees_key))
            mentors_key["group_index"] = range(len(mentors_key))


        #   Step 4: identify new matches by inner joining mentees_key and mentors_key by group_key and 
        #           group_index and add to the matching set 

        merged = pd.merge(
            mentees_key, 
            mentors_key,
            how="inner",
            on=["group_key", "group_index"],
            suffixes=["_mentee", "_mentor"]
        )       

        for _, row in merged.iterrows(): 
            
            #   Full row to compute the score of this specific match 
            mentee_row = mentees.loc[mentees["id"] == row["id_mentee"]].iloc[0]
            mentor_row = mentors.loc[mentors["id"] == row["id_mentor"]].iloc[0]

            #   Compute the matching score
            score = weights_score(mentee_row, mentor_row, weights)

            matching.append((row["id_mentee"], row["id_mentor"], score, iteration))


        #   Step 5 and 6: identify unmatched mentees and mentors

        matched_mentees = set(merged["id_mentee"]) 
        matched_mentors = set(merged["id_mentor"])

        unmatched_mentees = unmatched_mentees[~unmatched_mentees["id"].isin(matched_mentees)]
        unmatched_mentors = unmatched_mentors[~unmatched_mentors["id"].isin(matched_mentors)]

    total_score = sum(x[2] for x in matching)
    computation_time = time.time() - start 
    return total_score, matching, computation_time




# ================================================================
# SIMPLE EXAMPLE: EVALUATION OF MATCHING ALGORITHMS ON SMALL DATA
# ================================================================
#
# Algorithms tested:
# - LP1: Linear programming with all attribute weights (gender, type, programme, subject)
# - LP2: Linear programming focusing only on gender and type
# - RBi: Rule-based heuristic (6-step version)
# - RBii: Rule-based heuristic (2-step simplified version)
# 
# For each method, we compute and print:
# - Matching pairs
# - Total score R^W (sum of weights)
# - Mean and minimum score across matches
# - Objective value Z for LP models
# - Computation time
# - Final summary table for compariso


if __name__ == "__main__":

    print("\n================== SIMPLE EXAMPLE ==================\n")

    # --- Linear Programming (LP1) ---
    Z_LP1, matching_LP1, time_LP1 = run_linear_programming(mentees_se_df, mentors_se_df, weights_LP1)
    total_score_LP1 = compute_total_score(matching_LP1)
    mean_score_LP1 = compute_mean_score(matching_LP1, n_mentees=6)
    min_score_LP1 = compute_min_score(matching_LP1)
    print(f"\nMethod: Linear Programming (variation 1)")
    print(f"Optimal objective value Z = {Z_LP1:.2f}")
    print(f"Total score R^W = {total_score_LP1:.2f}")
    print(f"Computation time: {time_LP1:.4f} s")
    for mentee_id, mentor_id, score in matching_LP1:
        print(f"  Mentee {mentee_id} matched with Mentor {mentor_id} with score = {score:.2f}")

    # --- Linear Programming (LP2) ---
    Z_LP2, matching_LP2, time_LP2 = run_linear_programming(mentees_se_df, mentors_se_df, weights_LP2)
    total_score_LP2 = compute_total_score(matching_LP2)
    mean_score_LP2 = compute_mean_score(matching_LP2, n_mentees=6)
    min_score_LP2 = compute_min_score(matching_LP2)
    print(f"\nMethod: Linear Programming (variation 2)")
    print(f"\nOptimal objective value Z = {Z_LP2:.2f}")
    print(f"Total score R^W = {total_score_LP2:.2f}")
    print(f"Computation time: {time_LP2:.4f} s")
    for mentee_id, mentor_id, score in matching_LP2:
        print(f"  Mentee {mentee_id} matched with Mentor {mentor_id} with score = {score:.2f}")

    # --- Rule-Based Heuristic (RBi - 6 steps) ---
    total_score_h1, matching_h1, time_h1 = run_heuristic(
        mentees=mentees_se_df, 
        mentors=mentors_se_df, 
        matching_criteria=matching_criteria_1, 
        weights=weights_LP1, 
        steps_with_filtering=steps_with_filtering_1
    )
    mean_score_h1 = compute_mean_score(matching_h1, n_mentees=6)
    min_score_h1 = compute_min_score(matching_h1)
    print(f"\nMethod: Rule-based heuristic (variation 1)")
    print(f"Total score R^W = {total_score_h1:.2f}")
    print(f"Computation time: {time_h1:.4f} s")
    for mentee_id, mentor_id, score, step in matching_h1:
        print(f"  Mentee {mentee_id} matched with Mentor {mentor_id} with score = {score:.2f} at iteration {step}")

    # --- Rule-Based Heuristic (RBii - 2 steps) ---
    print(f"\nMethod: Rule-based heuristic (variation 2)")
    total_score_h2, matching_h2, time_h2 = run_heuristic(
        mentees=mentees_se_df, 
        mentors=mentors_se_df, 
        matching_criteria=matching_criteria_2, 
        weights=weights_LP1
    )
    mean_score_h2 = compute_mean_score(matching_h2, n_mentees=6)
    min_score_h2 = compute_min_score(matching_h2)
    print(f"\nMethod: Rule-based heuristic (variation 2)")
    print(f"Total score R^W = {total_score_h2:.2f}")
    print(f"Computation time: {time_h2:.4f} s")
    for mentee_id, mentor_id, score, step in matching_h2:
        print(f"  Mentee {mentee_id} matched with Mentor {mentor_id} with score = {score:.2f} at iteration {step}")

    # --- Summary Table ---
    print("\n================== SUMMARY TABLE ==================\n")
    print(f"{'ALGORITHM':<10} {'Total score':>12} {'Mean score':>12} {'Min score':>12} {'Z':>8} {'TIME (s)':>10}")
    print("-"*66)
    print(f"{'LP1':<10} {total_score_LP1:12.2f} {mean_score_LP1:12.2f} {min_score_LP1:12.2f} {Z_LP1:8.2f} {time_LP1:10.4f}")
    print(f"{'LP2':<10} {total_score_LP2:12.2f} {mean_score_LP2:12.2f} {min_score_LP2:12.2f} {Z_LP2:8.2f} {time_LP2:10.4f}")
    print(f"{'RBi':<10} {total_score_h1:12.2f} {mean_score_h1:12.2f} {min_score_h1:12.2f} {'-':>8} {time_h1:10.4f}")
    print(f"{'RBii':<10} {total_score_h2:12.2f} {mean_score_h2:12.2f} {min_score_h2:12.2f} {'-':>8} {time_h2:10.4f}")
