# 🎯 Project 3: AI Recommendation Logic


## 📌 Overview

This project implements a **Tech Stack Recommender** using **Content-Based Filtering**. It takes a user's current skills as input and recommends the most suitable career paths by computing similarity scores using **TF-IDF vectorization** and **Cosine Similarity** — no external ML libraries required, built from scratch using pure Python.

---

## 🎯 Goal

Build an AI-powered recommendation engine that analyzes a user's skill set and suggests the top 3 matching career paths from a dataset of 20 job roles.

---

## 📁 Project Structure

```
Decodelabs Internship/
│
└── WEEK 3/
    ├── recommender.py       # Main recommender engine
    ├── raw_skills.csv       # Dataset (20 job roles + skills)
    └── README.md            # Project documentation
```

---

## 🤖 Project Details

| Property       | Details                            |
|----------------|------------------------------------|
| Algorithm      | Content-Based Filtering            |
| Vectorizer     | TF-IDF (from scratch)              |
| Similarity     | Cosine Similarity (from scratch)   |
| Dataset        | raw_skills.csv (20 job roles)      |
| Language       | Python 3                           |
| Libraries      | csv, math, os (all built-in)       |
| Interface      | Terminal / Command Line            |

---

## ⚙️ Tech Stack

| Tool         | Purpose                              |
|--------------|--------------------------------------|
| Python 3     | Programming language                 |
| csv module   | Reading the job roles dataset        |
| math module  | Computing TF-IDF and cosine formula  |
| os module    | Locating the CSV file path           |

---

## 🔁 Pipeline: IPO Framework

```
INPUT                  →        PROCESS               →      OUTPUT
────────────────────────────────────────────────────────────────────
User Skills                  STEP 1: Ingestion              Top 3
raw_skills.csv               → Build Vocabulary          Career Paths
                             → Compute TF-IDF            + Similarity
                             STEP 2: Scoring               Score Bar
                             → Cosine Similarity
                             STEP 3: Sorting
                             → Rank by Score
                             STEP 4: Filtering
                             → Return Top 3
```

### The 4-Step Engine

**STEP 1 — Ingestion**
- Loads all job roles and skills from `raw_skills.csv`
- Builds a shared vocabulary from all skills
- Computes TF (Term Frequency) and IDF (Inverse Document Frequency)
- Converts every job role and user input into a TF-IDF weighted vector

**STEP 2 — Scoring**
- Computes Cosine Similarity between the user vector and every job role vector
- Formula: `cos(θ) = (A · B) / (||A|| × ||B||)`
- Returns a score between 0 and 1 for each role

**STEP 3 — Sorting**
- Sorts all job roles by their similarity score in descending order

**STEP 4 — Filtering**
- Returns only the Top 3 recommended career paths

---

## 📊 Dataset: raw_skills.csv

| Property     | Details                  |
|--------------|--------------------------|
| File         | raw_skills.csv           |
| Job Roles    | 20                       |
| Format       | job_role, skills         |
| Skill Format | Lowercase, underscore    |

**Sample roles included:** Data Scientist, ML Engineer, Full Stack Developer, Frontend Developer, Backend Developer, Computer Vision Engineer, DevOps Engineer, and more.

---

## 💡 Key Concepts Used

**TF-IDF (Term Frequency — Inverse Document Frequency)**
- TF: How often a skill appears in a role's skill list
- IDF: How rare that skill is across all roles
- Together: Gives more weight to unique, specific skills

**Cosine Similarity**
- Measures the angle between two skill vectors
- Score of 1.0 = perfect match, 0.0 = no match
- Ignores vector magnitude, focuses on direction (skill overlap)

**Content-Based Filtering**
- Recommends based on the content (skills) of the user profile
- No user history or ratings needed
- Pure skill-to-role matching

**Cold Start Handling**
- If none of the user's skills match the vocabulary, a warning is shown with suggested skill examples

---

## 🚀 How to Run

### 1. Make sure Python is installed
```bash
python --version
```

### 2. Navigate to the project folder
```bash
cd "Decodelabs Internship/WEEK 3"
```

### 3. Run the recommender
```bash
python recommender.py
```

### 4. Enter your skills
```
============================================================
   🤖  TECH STACK RECOMMENDER — DecodeLabs Project 3
============================================================
  Algorithm : TF-IDF + Cosine Similarity
  Type      : Content-Based Filtering
============================================================

  ✅ Loading dataset...
  📊 Loaded 20 job roles from raw_skills.csv

  Enter your skills one by one.
  Use underscores for multi-word skills (e.g., Machine_Learning)
  Press ENTER with no input when done (minimum 3 skills).

  Skill #1: python
  Skill #2: machine_learning
  Skill #3: sql
  Skill #4:

  📥 Your Skills : python, machine_learning, sql
  ────────────────────────────────────────────────────────────
  🏆 TOP RECOMMENDED CAREER PATHS
  ────────────────────────────────────────────────────────────
  🥇  Data Scientist        [█████████████░░░░░░░░░░░░░░░░░] 66.20%
  🥈  ML Engineer           [████████░░░░░░░░░░░░░░░░░░░░░░] 35.20%
  🥉  Computer Vision Eng.  [███░░░░░░░░░░░░░░░░░░░░░░░░░░░] 12.08%
  ────────────────────────────────────────────────────────────
```

---

## 🧪 Sample Test Results

### Test 1 — Web Development Skills
| Input Skills | java, html, css, sql, nodejs |
|---|---|
| 🥇 Recommendation | Full Stack Developer — 50.77% |
| 🥈 Recommendation | Frontend Developer — 35.69% |
| 🥉 Recommendation | Backend Developer — 25.68% |

### Test 2 — AI/ML Skills
| Input Skills | python, machine_learning, artificial_intelligence, sql, pandas, numpy |
|---|---|
| 🥇 Recommendation | Data Scientist — 66.20% |
| 🥈 Recommendation | ML Engineer — 35.20% |
| 🥉 Recommendation | Computer Vision Engineer — 12.08% |

---

## ⚠️ Cold Start Warning

If you enter skills that don't match the vocabulary:
```
⚠️  COLD START DETECTED
   None of your skills matched our vocabulary.
   Try skills like: Python, SQL, AWS, Docker, Java,
   Machine_Learning, React, Kubernetes, TensorFlow...
```

---

## 🆚 Project Comparison

| Feature          | Project 1 (Chatbot) | Project 2 (KNN)     | Project 3 (Recommender) |
|------------------|---------------------|---------------------|--------------------------|
| Type             | Rule-Based AI       | Supervised Learning | Content-Based Filtering  |
| Algorithm        | if-else logic       | K-Nearest Neighbors | TF-IDF + Cosine Sim      |
| Libraries        | None                | scikit-learn        | None (pure Python)       |
| Input            | Text commands       | Flower measurements | User skill list          |
| Output           | Chat response       | Flower species      | Top 3 career paths       |
| Complexity       | Beginner            | Intermediate        | Intermediate             |

---