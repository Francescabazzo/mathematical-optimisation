"""
dataset_generator.py

This module generates random datasets of mentees and mentors, based on the distributions described 
in the paper "Comparison of a rule-based heuristic and a linear programming model".

Three distributions are implemented, each characterized by different distributions for the mentee and
mentor attributes (such as gender, programme, type...):


    The following distributions don't change between the different scenarios: 

    programme_dist = [("CIS", 0.33), ("Sc", 0.33), ("Eng", 0.34)]
    subjects = {
        "CIS": ["d", "e", "f"],
        "Sc": ["a", "b", "c", "d"],
        "Eng": ["f", "g", "h"]
    }


    Distribution 1

    mentee_gender_pref_dist = [("female", 0.5), ("male", 0.4), ("no_pref", 0.1)]
    mentee_type_dist = [(1, 0.5), (0, 0.5)]

    mentor_gender_dist = [("female", 0.5), ("male", 0.5)]
    mentor_type_dist = [(1, 0.5), (0, 0.5)]


    Distribution 2 

    mentee_gender_pref_dist = [("female", 0.8), ("male", 0.1), ("other", 0.01) ("no_pref", 0.09)]
    mentee_type_dist = [(1, 0.5), (0, 0.5)]

    mentor_gender_dist = [("female", 0.7), ("male", 0.29), ("other", 0.01)]
    mentor_type_dist = [(1, 0.5), (0, 0.5)]


    Distribution 3 

    mentee_gender_pref_dist = [("female", 0.8), ("male", 0.1), ("other", 0.01) ("no_pref", 0.09)]
    mentee_type_dist = [(1, 0.5), (0, 0.5)]

    mentor_gender_dist = [("female", 0.7), ("male", 0.29), ("other", 0.01)]
    mentor_type_dist = [(1, 0.5), (0, 0.5)]

"""

import csv 
import random 
import pandas as pd



def weighted_choice(choices):
    """
    Randomly selects an element from a list of (value, weight) tuples 

    Parameters  
    -------
        choices: list of tuples
            Each tuple contain (value, weight)

    Returns
    -------
        Randomly generated value according to the weights.
    """
    values, weights = zip(*choices)
    return random.choices(values, weights=weights, k=1)[0]


def generate_dataset(num_mentees=100, num_mentors=100, scenario=1, save=False): 
    """
    Generates a dataset of mentees and mentors according to one of the three scenarios
    described in the paper.

    Parameters
    -------
        num_mentees: int, optional
            Number of mentees to generate, the default is 100.
        num_mentors: int, optional 
            Numbers of mentors to generate, the defauls is 100. 
        scenario: int, optional 
            Scenario number (1, 2, or 3), it determines the distributions to generate attributes. 
        save: bool, optional 
            If True, saves the generated datasets as 'mentees.csv' and 'mentors.csv' in the current directory.

    Returns
        -------
        mentees_df : pandas.DataFrame
            DataFrame containing the generated mentees with columns:
            ['id', 'gender_pref', 'programme', 'subject', 'type']
        mentors_df : pandas.DataFrame
            DataFrame containing the generated mentors with columns:
            ['id', 'gender', 'programme', 'subject', 'type'
    """
    if scenario == 2: 
        mentee_gender_pref_dist = [("female", 0.8), ("male", 0.1), ("other", 0.01), ("no_pref", 0.09)]
        mentee_type_dist = [(1, 0.5), (0, 0.5)]
        mentor_gender_dist = [("female", 0.7), ("male", 0.29), ("other", 0.01)]
        mentor_type_dist = [(1, 0.5), (0, 0.5)]
    elif scenario == 3:
        mentee_gender_pref_dist = [("female", 0.8), ("male", 0.1), ("other", 0.01), ("no_pref", 0.09)]
        mentee_type_dist = [(1, 0.5), (0, 0.5)]
        mentor_gender_dist = [("female", 0.7), ("male", 0.29), ("other", 0.01)]
        mentor_type_dist = [(1, 0.5), (0, 0.5)]
    else:   #   For scenario == 1 and for all the other inputs 
        mentee_gender_pref_dist = [("female", 0.5), ("male", 0.4), ("no_pref", 0.1)]
        mentee_type_dist = [(1, 0.5), (0, 0.5)]
        mentor_gender_dist = [("female", 0.5), ("male", 0.5)]
        mentor_type_dist = [(1, 0.5), (0, 0.5)]
    
    #   The programmes and the subjects' distributions are the same for all the scenarios.
    programme_dist = [("CIS", 0.33), ("Sc", 0.33), ("Eng", 0.34)]
    subjects = {
        "CIS": ["d", "e", "f"],
        "Sc": ["a", "b", "c", "d"],
        "Eng": ["f", "g", "h"]
    }

    # Generate mentees
    mentees = []
    for i in range(num_mentees):
        mentee_id = f"M{i+1:03}"
        gender_pref = weighted_choice(mentee_gender_pref_dist)
        type_ = weighted_choice(mentee_type_dist)
        programme = weighted_choice(programme_dist)
        subject = random.choice(subjects[programme])
        mentees.append([mentee_id, gender_pref, programme, subject, type_])

    # Generate mentors
    mentors = []
    for i in range(num_mentors):
        mentor_id = f"T{i+1:03}"
        gender = weighted_choice(mentor_gender_dist)
        type_ = weighted_choice(mentor_type_dist)
        
        # a mentor can have 1 or 2 programs
        num_programmes = random.choice([1,2])
        mentor_programmes = random.sample(["CIS", "Sc", "Eng"], num_programmes)

        # pick preferred programme
        preferred_programme = mentor_programmes[0]
        
        # for the subject, I take from all the ones defined 
        mentor_subjects = []
        for prog in mentor_programmes:
            mentor_subjects += random.sample(subjects[prog], k=random.choice([1,2]))
        # remove duplicates
        mentor_subjects = list(set(mentor_subjects))
        
        mentors.append([
            mentor_id, 
            gender, 
            ";".join(mentor_programmes), 
            preferred_programme,
            ";".join(mentor_subjects), 
            type_
            ])


    #   Define the dataframes 

    mentees_df = pd.DataFrame(mentees, columns=["id", "gender_pref", "programme", "subject", "type"])
    mentors_df = pd.DataFrame(mentors, columns=["id", "gender", "programme", "preferred_programme", "subject", "type"])

    #   If save is True, save the dataframes as cvs files in the disk 

    if save==True: 

        # Save mentees.csv
        with open("mentees.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "gender_pref", "programme", "subject", "type"])
            writer.writerows(mentees)

        # Save mentors.csv
        with open("mentors.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "gender", "programme", "preferred_program", "subject", "type"])
            writer.writerows(mentors)

        print("Files mentees.csv and mentors.csv generated")

    
    #   Return the two dataframes 

    return mentees_df, mentors_df 