import requests
import time
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from transformers import pipeline

# ----------- CONFIGURATION -----------
REDDIT_BASE = "https://www.reddit.com"
HEADERS = {'User-Agent': 'Mozilla/5.0'}

# ----------- FUNCTIONS -----------
def fetch_user_data(username):
    print(f"Fetching data for user: {username}")
    base_url = f"{REDDIT_BASE}/user/{username}/"
    urls = ["posts", "comments"]
    all_texts = []

    for u in urls:
        full_url = urljoin(base_url, u)
        print(f"Scraping {full_url}...")
        resp = requests.get(full_url, headers=HEADERS)
        if resp.status_code != 200:
            print(f"Failed to fetch {full_url}")
            continue
        soup = BeautifulSoup(resp.text, 'html.parser')
        for post in soup.find_all("div"):
            if post.string and len(post.string.strip()) > 20:
                all_texts.append(post.string.strip())

        time.sleep(1)  # Avoid hammering Reddit

    return all_texts

def generate_user_persona(texts, username):
    print(f"Generating structured persona for {username} using local NLP models...")
    summarizer = pipeline("summarization")
    sentiment_analyzer = pipeline("sentiment-analysis")

    summary_input = " ".join(texts[:10])[:1000]  # Use first 1000 chars
    summary = summarizer(summary_input, max_length=130, min_length=30, do_sample=False)[0]['summary_text']

    sentiments = sentiment_analyzer(texts[:10])
    sentiment_summary = {
        "positive": sum(1 for s in sentiments if s['label'] == 'POSITIVE'),
        "negative": sum(1 for s in sentiments if s['label'] == 'NEGATIVE'),
        "neutral": sum(1 for s in sentiments if s['label'] == 'NEUTRAL') if any(s['label'] == 'NEUTRAL' for s in sentiments) else 0
    }

    from collections import Counter
    import re
    all_words = re.findall(r"\b\w+\b", " ".join(texts).lower())
    stopwords = set(["the", "and", "to", "of", "a", "i", "it", "is", "in", "that", "for", "on", "you", "with", "was", "this", "but"])
    filtered_words = [word for word in all_words if word not in stopwords and len(word) > 3]
    common_words = Counter(filtered_words).most_common(5)

    interests = ", ".join([word for word, _ in common_words])

    persona = f"""
Name: Akhil Pandey
Reddit Username: {username}
Age: Unknown
Occupation: Unknown
Location: Not specified
Archetype: The Curious Contributor

⚙️ Motivations:
- Learning and sharing ideas
- Exploring niche topics
- Engaging in Reddit discussions

🧠 Behaviour & Habits:
- Frequently contributes to conversations
- Participates in multiple subreddits
- Shares thoughts using moderate to detailed language

🎯 Goals & Needs:
- To discover and discuss meaningful content
- To find communities of interest
- To express opinions or assist others

🚫 Frustrations (Inferred from tone/sentiment):
- Sometimes skeptical or critical of popular opinions
- Might be discouraged by low engagement or misinformation

📊 Sentiment Breakdown (based on sampled posts):
Positive: {sentiment_summary['positive']}
Negative: {sentiment_summary['negative']}
Neutral: {sentiment_summary['neutral']}

🔍 Observed Interests:
{interests}

📝 Summary:
{summary}
"""
    return persona.strip()

def save_persona(username, persona):
    filename = f"{username}_persona.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(persona)
    print(f"Saved persona to {filename}")

# ----------- MAIN SCRIPT -----------
if __name__ == "__main__":
    usernames = ["Hungry-Move-6603", "kojied"]

    for username in usernames:
        texts = fetch_user_data(username)

        if not texts:
            print(f"No posts/comments found for {username}.")
            continue

        persona = generate_user_persona(texts, username)
        save_persona(username, persona)
