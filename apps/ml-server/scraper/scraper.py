import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
from textblob import TextBlob
import re
import time
import os
import queue
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download NLTK data
# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    # Add a check for punkt_tab
    nltk.data.find('tokenizers/punkt_tab')
except:
    nltk.download('punkt')
    nltk.download('stopwords')
    # Download the missing resource
    nltk.download('punkt_tab')
# ------------------------------
# Load .env
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found in .env")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in env")

youtube = build("youtube", "v3", developerKey=API_KEY)
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(api_version='v1alpha')
)

# ------------------------------
# Helper functions - OPTIMIZED
# ------------------------------

def extract_video_id(url):
    parsed_url = urlparse(url)
    video_id = parse_qs(parsed_url.query).get("v")
    if video_id:
        return video_id[0]
    if "youtu.be" in parsed_url.netloc:
        return parsed_url.path.strip("/")
    return None

def clean_text(text):
    if not text:
        return ""
    text = re.sub(r"[_*~`#>!\[\]\(\)-]", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_keywords(text, max_keywords=20):
    """Extract keywords from description using TF-IDF"""
    if not text or len(text.split()) < 5:
        return []
    
    # Preprocess text
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalpha() and word not in stop_words and len(word) > 2]
    
    if len(filtered_words) < 3:
        return []
    
    # Use simple frequency-based approach (faster than TF-IDF for single docs)
    word_freq = {}
    for word in filtered_words:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top keywords by frequency
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
    return [word for word, freq in keywords]

def analyze_sentiment(comments):
    if not comments:
        return {"positive": 0, "neutral": 0, "negative": 0}
    pos, neu, neg = 0, 0, 0
    for c in comments:
        try:
            polarity = TextBlob(c).sentiment.polarity
            if polarity > 0.1:
                pos += 1
            elif polarity < -0.1:
                neg += 1
            else:
                neu += 1
        except:
            continue
    
    total = pos + neu + neg
    if total == 0:
        return {"positive": 0, "neutral": 0, "negative": 0}
    
    return {
        "positive": round((pos / total) * 100, 2),
        "neutral": round((neu / total) * 100, 2),
        "negative": round((neg / total) * 100, 2),
    }

def categorize_difficulty(title, description, keywords):
    """Fast difficulty categorization using description only"""
    text = (title + " " + description + " " + " ".join(keywords)).lower()
    
    # Fast heuristic approach first
    beginner_indicators = ['beginner', 'basic', 'intro', 'introduction', 'fundamental', 'starter', 'crash course', 'overview']
    advanced_indicators = ['advanced', 'expert', 'master', 'deep dive', 'complex', 'sophisticated', 'in-depth']
    
    if any(indicator in text for indicator in beginner_indicators):
        return "Beginner"
    elif any(indicator in text for indicator in advanced_indicators):
        return "Advanced"
    else:
        # Fallback to Gemini only when needed
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=f"Classify as Beginner, Intermediate, or Advanced: {text[:1000]}"
            )
            return response.text.strip()
        except:
            return "Intermediate"

def get_comments(video_id, max_comments=15):
    """Get comments with error handling"""
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_comments,
            textFormat="plainText"
        )
        response = request.execute()
        comments = [
            item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            for item in response.get("items", [])
        ]
        return comments
    except Exception as e:
        print(f"Comments error for {video_id}: {e}")
        return []

def get_video_data(video_url, topic):
    """FAST version - no transcript, description only"""
    try:
        video_id = extract_video_id(video_url)
        if not video_id:
            return None

        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        )
        response = request.execute()
        if not response.get("items"):
            return None

        video_info = response["items"][0]
        snippet = video_info["snippet"]
        stats = video_info.get("statistics", {})
        content = video_info.get("contentDetails", {})

        title = clean_text(snippet.get("title", ""))
        description = clean_text(snippet.get("description", ""))

        views = stats.get("viewCount", "0")
        likes = stats.get("likeCount", "0")
        length = content.get("duration", "PT0M0S")  # ISO 8601 duration
        
        # Convert duration to minutes
        duration_minutes = parse_duration(length)

        # Extract keywords from description (FAST - no transcript)
        keywords = extract_keywords(description)
        
        # Comments + sentiment
        comments = get_comments(video_id)
        sentiment = analyze_sentiment(comments)

        # Difficulty
        difficulty = categorize_difficulty(title, description, keywords)

        return {
            "topic": topic,
            "video_id": video_id,
            "title": title,
            "description": description[:500],  # Store first 500 chars
            "views": int(views),
            "likes": int(likes) if likes else 0,
            "duration_minutes": duration_minutes,
            "keywords": ", ".join(keywords) if keywords else "No keywords",
            "keyword_count": len(keywords),
            "difficulty": difficulty,
            "positive_comments(%)": sentiment["positive"],
            "neutral_comments(%)": sentiment["neutral"],
            "negative_comments(%)": sentiment["negative"],
            "total_comments": len(comments),
            "video_url": video_url,
            "published_at": snippet.get("publishedAt", ""),
            "channel_title": snippet.get("channelTitle", "")
        }
    except Exception as e:
        print(f"Error processing {video_url}: {e}")
        return None

def parse_duration(duration_str):
    """Convert ISO 8601 duration to minutes"""
    # Example: "PT1H30M15S" -> 90.25 minutes
    try:
        import isodate
        duration = isodate.parse_duration(duration_str)
        return duration.total_seconds() / 60
    except:
        # Fallback: simple regex parsing
        hours = re.search(r'(\d+)H', duration_str)
        minutes = re.search(r'(\d+)M', duration_str)
        seconds = re.search(r'(\d+)S', duration_str)
        
        total_minutes = 0
        if hours:
            total_minutes += int(hours.group(1)) * 60
        if minutes:
            total_minutes += int(minutes.group(1))
        if seconds:
            total_minutes += int(seconds.group(1)) / 60
            
        return round(total_minutes, 2)

def search_videos(query, max_results=25):
    try:
        request = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=max_results,
            order="viewCount"  # Get popular videos first
        )
        response = request.execute()
        return [
            f"https://www.youtube.com/watch?v={item['id']['videoId']}"
            for item in response.get("items", [])
        ]
    except Exception as e:
        print(f"Error searching videos: {e}")
        return []

# ------------------------------
# FAST Pipeline
# ------------------------------

def fast_pipeline(topics, max_results=25):
    job_queue = queue.Queue()

    # enqueue jobs
    for topic in topics:
        video_urls = search_videos(topic, max_results=max_results)
        for url in video_urls:
            job_queue.put((url, topic))

    all_data = []
    processed_count = 0

    # process jobs
    while not job_queue.empty():
        url, topic = job_queue.get()
        print(f"📥 Processing ({processed_count+1}): {topic} -> {url}")
        video_data = get_video_data(url, topic)
        if video_data:
            all_data.append(video_data)
            processed_count += 1

        # Respect API rate limits
        time.sleep(random.uniform(0.1, 0.3))  # Much faster now!

    # Save dataset
    filename = os.path.join(BASE_DIR, "fast_dataset.csv")
    df = pd.DataFrame(all_data)

    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"✅ Saved {len(all_data)} videos to {filename}")
    
    return df

# ------------------------------
# Run
# ------------------------------

if __name__ == "__main__":
    TOPICS = [
        "react js beginner", "react js advanced",
        "machine learning basics", "machine learning advanced",
        "database management system", "operating system concepts",
        "data structures and algorithms", "computer networks",
        "cybersecurity basics", "software engineering",
        "artificial intelligence basics", "cloud computing",
        "devops docker kubernetes", "web development html css javascript",
        "backend development rest api", "mobile app development flutter"
    ]
    
    print("🚀 Starting FAST dataset creation (description only)...")
    df = fast_pipeline(TOPICS, max_results=50)
    
    print("\n📊 Dataset Summary:")
    print(f"Total videos: {len(df)}")
    print(f"Topics covered: {df['topic'].nunique()}")
    print(f"Difficulty distribution:")
    print(df['difficulty'].value_counts())