# Mentor-Mentee Matching Algorithms
_Project for the Mathematical Optimisation Exam_

---

## About this Project

This project was developed as part of the course "Mathematical Optimisation" (Master’s Degree of Computer Engineering, University of Trieste).  
It focuses on implementing and comparing different matching algorithms for mentor-mentee assignments, including:

- **Linear Programming** (LP1 and LP2)
- **Rule-Based Heuristics** (RBi and RBii)

It also performs scalability and performance analysis on various dataset sizes. 

---

## 📁 Project Structure

```
📦mathematicaloptimisation/
├── data/
│   ├── mentees_se.csv           # Small example mentees dataset
│   └── mentors_se.csv           # Small example mentors dataset
├── dataset_generator.py         # Random dataset generator
├── test.py                      # Run and compare LP and heuristic approaches on a small example
├── scalability.py               # Scalability analysis with increasing dataset size
├── numerical_experiments.py     # Large-scale performance comparison (fixed dataset size)
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/framcescabazzo/mathematicalotpimisation.git
cd matemathicaloptimisation
```

### 2. Install dependencies

Make sure you have **Python 3.9+** and **Gurobi** properly installed and activated (requires an academic license).

Install the Python packages:

```bash
pip install pandas numpy matplotlib
```

---

## ▶️ Running the Main Scripts

### ✅ Run `test.py` — Small Example

This script runs a small example using the provided datasets in `data/`. It computes and prints:

- Matchings using LP1, LP2, RBi, and RBii
- Total, mean, and minimum scores 
- Computation time for each algorithm
- A final summary table

To execute:

```bash
python test.py
```

---

### 📊 Run `numerical_experiments.py` — Large-Scale Performance

This script evaluates **12 algorithms** (6 LP variants and 6 rule-based heuristics) across **100 replications** on small, medium and large datasets.

It computes the **mean matching score** for each algorithm and provides a summary table (mean, median, std deviation).

To run:

```bash
python numerical_experiments.py
```

⚠️ **Note:** This script is computationally intensive and can take **several minutes** to complete.

---

### 📈 Run `scalability.py` — Scalability Analysis

This script tests the **scalability** of 4 key algorithms (LP1, LP2, RBi, RBii) by increasing dataset size (e.g., 10, 50, 100, 500, 1000).

It computes and plots:
- Total matching score 
- Mean score per mentee 
- Computation time vs dataset size

To run:

```bash
python scalability.py
```

It will generate:
- `scalability_results.csv`
- `scalability_times.png`
- `scalability_scores.png`
- `scalability_mean_scores.png`

⚠️ **Note:** This script is computationally intensive and can take **several minutes** to complete.


---

## 📌 Notes

- LP algorithms use Gurobi and may require proper licensing.
- All scores are computed using a consistent weight configuration (`weights_LP1`) for comparability, as explained in the reference paper.

---

## 📚 Reference

This project is based on methodologies inspired by the paper:

> *Marshall, S. E., & Mohaghegh, M. (2025). Comparison of a rule-based heuristic and a linear programming model for assigning mentees and mentors in a women in technology mentoring programme. Computers & Operations Research, 107002.*
