"""
========================================================
  Project 3: AI Recommendation Logic
  Tech Stack Recommender
  DecodeLabs Industrial Training Kit | Batch 2026
========================================================
  Algorithm  : Content-Based Filtering
  Vectorizer : TF-IDF (Term Frequency - Inverse Document Frequency)
  Similarity : Cosine Similarity
  Pipeline   : Ingestion → Scoring → Sorting → Filtering
========================================================
"""

import csv
import math
import os


# ─────────────────────────────────────────────
#  STEP 0: LOAD DATASET
# ─────────────────────────────────────────────

def load_dataset(filepath):
    """Load job roles and their skills from a CSV file."""
    dataset = {}
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = row['job_role'].strip()
            skills = row['skills'].strip().lower().split()
            dataset[role] = skills
    return dataset


# ─────────────────────────────────────────────
#  STEP 1: INGESTION — Build Vocabulary & Vectors
# ─────────────────────────────────────────────

def build_vocabulary(dataset):
    """Build a shared vocabulary from all job role skills."""
    vocab = set()
    for skills in dataset.values():
        vocab.update(skills)
    return sorted(vocab)


def compute_tf(skill_list):
    """Compute Term Frequency for a list of skills."""
    tf = {}
    total = len(skill_list)
    for skill in skill_list:
        tf[skill] = tf.get(skill, 0) + 1
    for skill in tf:
        tf[skill] = tf[skill] / total
    return tf


def compute_idf(dataset):
    """Compute Inverse Document Frequency across all job roles."""
    total_docs = len(dataset)
    doc_freq = {}
    for skills in dataset.values():
        for skill in set(skills):
            doc_freq[skill] = doc_freq.get(skill, 0) + 1
    idf = {}
    for skill, freq in doc_freq.items():
        idf[skill] = math.log(total_docs / freq)
    return idf


def build_tfidf_vector(skill_list, vocabulary, idf):
    """Build a TF-IDF weighted vector for a skill list."""
    tf = compute_tf(skill_list)
    vector = []
    for term in vocabulary:
        tf_val = tf.get(term, 0)
        idf_val = idf.get(term, 0)
        vector.append(tf_val * idf_val)
    return vector


def build_user_vector(user_skills, vocabulary, idf):
    """
    Build a TF-IDF vector for the user profile.
    User inputs are treated as a flat skill document.
    """
    normalized = [s.lower().replace(" ", "_") for s in user_skills]
    return build_tfidf_vector(normalized, vocabulary, idf)


# ─────────────────────────────────────────────
#  STEP 2: SCORING — Cosine Similarity
# ─────────────────────────────────────────────

def cosine_similarity(vec_a, vec_b):
    """
    Compute cosine similarity between two vectors.
    cos(θ) = (A · B) / (||A|| × ||B||)
    Returns a score between 0 and 1.
    """
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    magnitude_a = math.sqrt(sum(a ** 2 for a in vec_a))
    magnitude_b = math.sqrt(sum(b ** 2 for b in vec_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0  # Cold Start: zero vector → no similarity

    return dot_product / (magnitude_a * magnitude_b)


def score_all_roles(user_vector, role_vectors):
    """Score every job role against the user profile vector."""
    scores = {}
    for role, role_vector in role_vectors.items():
        scores[role] = cosine_similarity(user_vector, role_vector)
    return scores


# ─────────────────────────────────────────────
#  STEP 3: SORTING — Rank by Score
# ─────────────────────────────────────────────

def sort_scores(scores):
    """Sort job roles by similarity score in descending order."""
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────
#  STEP 4: FILTERING — Return Top-N Results
# ─────────────────────────────────────────────

def filter_top_n(sorted_scores, n=3):
    """Return only the top N recommendations."""
    return sorted_scores[:n]


# ─────────────────────────────────────────────
#  DISPLAY HELPERS
# ─────────────────────────────────────────────

def display_banner():
    print("=" * 60)
    print("   🤖  TECH STACK RECOMMENDER — DecodeLabs Project 3")
    print("=" * 60)
    print("  Algorithm : TF-IDF + Cosine Similarity")
    print("  Type      : Content-Based Filtering")
    print("=" * 60)
    print()


def display_recommendations(recommendations, user_skills):
    print()
    print("─" * 60)
    print(f"  📥 Your Skills  : {', '.join(user_skills)}")
    print("─" * 60)
    print("  🏆  TOP RECOMMENDED CAREER PATHS")
    print("─" * 60)

    medals = ["🥇", "🥈", "🥉"]
    for i, (role, score) in enumerate(recommendations):
        medal = medals[i] if i < len(medals) else f"#{i+1}"
        bar_length = int(score * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        print(f"  {medal}  {role:<30} [{bar}] {score:.2%}")

    print("─" * 60)
    print()


def display_cold_start_warning():
    print()
    print("⚠️  COLD START DETECTED")
    print("   None of your skills matched our vocabulary.")
    print("   Try skills like: Python, SQL, AWS, Docker, Java,")
    print("   Machine_Learning, React, Kubernetes, TensorFlow...")
    print()


# ─────────────────────────────────────────────
#  MAIN ENGINE
# ─────────────────────────────────────────────

def get_user_input():
    """Prompt the user to enter at least 3 skills."""
    print("  Enter your skills one by one.")
    print("  Use underscores for multi-word skills (e.g., Machine_Learning)")
    print("  Press ENTER with no input when done (minimum 3 skills).\n")

    skills = []
    while True:
        skill = input(f"  Skill #{len(skills) + 1}: ").strip()
        if skill == "":
            if len(skills) < 3:
                print(f"  ⚠️  Please enter at least {3 - len(skills)} more skill(s).\n")
            else:
                break
        elif skill:
            skills.append(skill)

    return skills


def run_recommender():
    display_banner()

    # Locate dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "raw_skills.csv")

    if not os.path.exists(csv_path):
        print(f"  ❌ Dataset not found at: {csv_path}")
        print("  Make sure raw_skills.csv is in the same folder as this script.")
        return

    # Load dataset
    print("  ✅ Loading dataset...\n")
    dataset = load_dataset(csv_path)
    print(f"  📊 Loaded {len(dataset)} job roles from raw_skills.csv\n")

    while True:
        # ── STEP 1: INGESTION ──────────────────────────
        user_skills = get_user_input()

        vocabulary = build_vocabulary(dataset)
        idf = compute_idf(dataset)

        # Build vectors for all job roles
        role_vectors = {
            role: build_tfidf_vector(skills, vocabulary, idf)
            for role, skills in dataset.items()
        }

        # Build user profile vector
        user_vector = build_user_vector(user_skills, vocabulary, idf)

        # Cold Start Check
        if all(v == 0 for v in user_vector):
            display_cold_start_warning()
        else:
            # ── STEP 2: SCORING ────────────────────────────
            scores = score_all_roles(user_vector, role_vectors)

            # ── STEP 3: SORTING ────────────────────────────
            sorted_scores = sort_scores(scores)

            # ── STEP 4: FILTERING ──────────────────────────
            top_recommendations = filter_top_n(sorted_scores, n=3)

            # ── OUTPUT ─────────────────────────────────────
            display_recommendations(top_recommendations, user_skills)

        # Ask if user wants to try again
        again = input("  🔄 Try with different skills? (y/n): ").strip().lower()
        print()
        if again != 'y':
            break

    print("  👋 Thank you for using the Tech Stack Recommender!")
    print("     DecodeLabs | Batch 2026\n")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    run_recommender()
