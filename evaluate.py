import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------- Load Data -----------------
with open("shl_assessments.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

with open("shl_test_queries.json", "r", encoding="utf-8") as f:
    test_queries = json.load(f)

# ----------------- Synonym Expansion -----------------
def expand_query(query):
    synonyms = {
        "coding": "coding programming code developer",
        "remote": "remote online virtual",
        "test": "test assessment evaluation exam",
        "manager": "manager lead supervisor",
        "sales": "sales marketing business",
        "communication": "communication verbal soft skills",
        "language": "language english verbal spoken"
    }

    words = query.lower().split()
    expanded = []

    for word in words:
        expanded.append(synonyms.get(word, word))

    return " ".join(expanded)

# ----------------- Prepare Vectorizer -----------------
def format_text(item):
    return f"{item['name']} {item['test_type']} duration {item['duration_minutes']} minutes " \
           f"remote {item['remote_support']} adaptive {item['adaptive_support']}"

texts = [format_text(item) for item in catalog]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# ----------------- Recommend -----------------
def get_recommendations(query, k=3):
    expanded = expand_query(query)
    query_vec = vectorizer.transform([expanded])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[-k:][::-1]
    return [catalog[i]["name"] for i in top_indices]

# ----------------- Metrics -----------------
def recall_at_k(expected, predicted):
    return len(set(expected) & set(predicted)) / len(expected)

def map_at_k(expected, predicted):
    score = 0.0
    hits = 0
    for i, p in enumerate(predicted[:3]):
        if p in expected:
            hits += 1
            score += hits / (i + 1)
    return score / min(len(expected), 3)

# ----------------- Evaluate -----------------
recalls = []
maps = []

for test in test_queries:
    predicted = get_recommendations(test["query"], 3)
    recalls.append(recall_at_k(test["expected"], predicted))
    maps.append(map_at_k(test["expected"], predicted))

print(f"📊 Mean Recall@3: {sum(recalls)/len(recalls):.3f}")
print(f"📊 Mean MAP@3:    {sum(maps)/len(maps):.3f}")
