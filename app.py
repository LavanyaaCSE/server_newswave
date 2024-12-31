from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
import feedparser
import requests

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load Tamil model, tokenizer, and label encoder
tamil_model = load_model('tamil_model.h5')
tamil_tokenizer = pickle.load(open('tamil_tokenizer.pkl', 'rb'))
tamil_label_encoder = pickle.load(open('tamil_label_encoder.pkl', 'rb'))

# Load English model, tokenizer, and label encoder
english_model = load_model('english_model.h5')
english_tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
english_label_encoder = pickle.load(open('label_encoder.pkl', 'rb'))

# Function to fetch RSS feed
def fetch_rss_feed(rss_url):
    return feedparser.parse(rss_url)

# Function to extract image URL from the RSS entry
def extract_image_url(entry):
    try:
        if hasattr(entry, "media_content") and entry.media_content:
            return entry.media_content[0]['url']
        if hasattr(entry, "enclosures") and entry.enclosures:
            return entry.enclosures[0]['href']
    except (AttributeError, KeyError, IndexError):
        pass
    return None

# Function to fetch city-specific RSS feed URL for Tamil news
def get_city_rss_url_tamil(city_name):
    if city_name.lower() == "chennai":
        return "https://tamil.news18.com/commonfeeds/v1/tam/rss/chennai-district.xml"
    elif city_name.lower() == "coimbatore":
        return "https://tamil.news18.com/commonfeeds/v1/tam/rss/coimbatore-district.xml"
    elif city_name.lower() == "madurai":
        return "https://tamil.news18.com/commonfeeds/v1/tam/rss/madurai-district.xml"
    elif city_name.lower() == "salem": 
        return "https://tamil.news18.com/commonfeeds/v1/tam/rss/salem-district.xml"
    elif city_name.lower() == "tiruppur":
        return "https://tamil.news18.com/commonfeeds/v1/tam/rss/tiruppur-district.xml"
    elif city_name.lower() == "tiruchirappalli":
        return "https://tamil.news18.com/commonfeeds/v1/tam/rss/tiruchirappalli-district.xml"
    elif city_name.lower() == "vellore":
        return "https://tamil.news18.com/commonfeeds/v1/tam/rss/vellore-district.xml"
    elif city_name.lower() == "thanjavur":
        return "https://tamil.news18.com/commonfeeds/v1/tam/rss/thanjavur-district.xml"
    elif city_name.lower() == "tirunelveli":
        return "https://tamil.news18.com/commonfeeds/v1/tam/rss/tirunelveli-district.xml"
    elif city_name.lower() == "kanchipuram":
        return "https://tamil.news18.com/commonfeeds/v1/tam/rss/kanchipuram-district.xml"
    return None

# Function to fetch city-specific RSS feed URL for English news
def get_city_rss_url_english(city_name):
    if city_name.lower() == "chennai":
        return "https://www.thehindu.com/news/cities/chennai/?service=rss"
    elif city_name.lower() == "coimbatore":
        return "https://www.thehindu.com/news/cities/coimbatore/?service=rss"
    elif city_name.lower() == "madurai":
        return "https://www.thehindu.com/news/cities/madurai/?service=rss"
    elif city_name.lower() == "salem":
        return "https://tamil.news18.com/commonfeeds/v1/tam/rss/salem-district.xml"
    elif city_name.lower() == "tiruppur":
        return "https://tamil.news18.com/commonfeeds/v1/tam/rss/tiruppur-district.xml"
    elif city_name.lower() == "tiruchirappalli":
        return "https://tamil.news18.com/commonfeeds/v1/tam/rss/tiruchirappalli-district.xml"
    elif city_name.lower() == "vellore":
        return "https://tamil.news18.com/commonfeeds/v1/tam/rss/vellore-district.xml"
    elif city_name.lower() == "thanjavur":
        return "https://tamil.news18.com/commonfeeds/v1/tam/rss/thanjavur-district.xml"
    elif city_name.lower() == "tirunelveli":
        return "https://tamil.news18.com/commonfeeds/v1/tam/rss/tirunelveli-district.xml"
    elif city_name.lower() == "kanchipuram":
        return "https://tamil.news18.com/commonfeeds/v1/tam/rss/kanchipuram-district.xml"
    return None

# Function to classify Tamil news
def classify_tamil_news(news_list):
    try:
        sequences = tamil_tokenizer.texts_to_sequences(news_list)
        padded_sequences = pad_sequences(sequences, maxlen=100)
        prediction = tamil_model.predict(padded_sequences)
        predicted_labels = tamil_label_encoder.inverse_transform(np.argmax(prediction, axis=1))

        categorized_news = {}
        for i, label in enumerate(predicted_labels):
            if label not in categorized_news:
                categorized_news[label] = []
            categorized_news[label].append(news_list[i])

        return categorized_news
    except Exception as e:
        raise RuntimeError(f"Classification error: {str(e)}")

# Function to classify English news
def classify_english_news(news_list):
    try:
        sequences = english_tokenizer.texts_to_sequences(news_list)
        padded_sequences = pad_sequences(sequences, maxlen=100)
        prediction = english_model.predict(padded_sequences)
        predicted_labels = english_label_encoder.inverse_transform(np.argmax(prediction, axis=1))

        categorized_news = {}
        for i, label in enumerate(predicted_labels):
            if label not in categorized_news:
                categorized_news[label] = []
            categorized_news[label].append(news_list[i])

        return categorized_news
    except Exception as e:
        raise RuntimeError(f"Classification error: {str(e)}")

@app.route('/get_tamil_city_news', methods=['POST'])
def get_tamil_city_news():
    city_name = request.json.get('city')
    if not city_name:
        return jsonify({'error': 'City name not provided'}), 400

    try:
        rss_url = get_city_rss_url_tamil(city_name)
        if not rss_url:
            return jsonify({'error': 'City not supported for Tamil news'}), 404
        
        feed = fetch_rss_feed(rss_url)

        news_articles = []
        seen_summaries = set()
        for entry in feed.entries:
            image_url = extract_image_url(entry)
            summary = entry.summary.strip()
            if summary not in seen_summaries:
                seen_summaries.add(summary)
                news_articles.append({
                    'title': entry.title,
                    'link': entry.link,
                    'summary': summary,
                    'published': entry.published,
                    'image_url': image_url
                })

        summaries = [article['summary'] for article in news_articles]
        categorized_news = classify_tamil_news(summaries)

        categorized_news_with_details = {}
        for category, news_list in categorized_news.items():
            categorized_news_with_details[category] = []
            for summary in news_list:
                article_details = next((article for article in news_articles if article['summary'] == summary), None)
                if article_details:
                    categorized_news_with_details[category].append(article_details)

        return jsonify(categorized_news_with_details)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_english_city_news', methods=['POST'])
def get_english_city_news():
    city_name = request.json.get('city')
    if not city_name:
        return jsonify({'error': 'City name not provided'}), 400

    try:
        rss_url = get_city_rss_url_english(city_name)
        if not rss_url:
            return jsonify({'error': 'City not supported for English news'}), 404
        
        feed = fetch_rss_feed(rss_url)

        news_articles = []
        seen_summaries = set()
        for entry in feed.entries:
            image_url = extract_image_url(entry)
            summary = entry.summary.strip()
            if summary not in seen_summaries:
                seen_summaries.add(summary)
                news_articles.append({
                    'title': entry.title,
                    'link': entry.link,
                    'summary': summary,
                    'published': entry.published,
                    'image_url': image_url
                })

        summaries = [article['summary'] for article in news_articles]
        categorized_news = classify_english_news(summaries)

        categorized_news_with_details = {}
        for category, news_list in categorized_news.items():
            categorized_news_with_details[category] = []
            for summary in news_list:
                article_details = next((article for article in news_articles if article['summary'] == summary), None)
                if article_details:
                    categorized_news_with_details[category].append(article_details)

        return jsonify(categorized_news_with_details)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_news', methods=['GET'])
def get_news():
    # Define RSS feed URLs for predefined categories
    rss_urls = [
        'https://www.news18.com/commonfeeds/v1/eng/rss/lifestyle-2.xml',
        'https://www.news18.com/commonfeeds/v1/eng/rss/movies.xml'
    ]
    
    try:
        all_news_articles = []
        
        # Fetch and parse each RSS feed
        for rss_url in rss_urls:
            feed = fetch_rss_feed(rss_url)
            
            for entry in feed.entries:
                image_url = extract_image_url(entry)
                summary = entry.summary.strip()
                all_news_articles.append({
                    'title': entry.title,
                    'link': entry.link,
                    'summary': summary,
                    'published': entry.published,
                    'image_url': image_url
                })
        
        # Return the list of news articles from all feeds
        return jsonify(all_news_articles)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_sun_news', methods=['GET'])
def get_sun_news():
    rss_url = 'https://fetchrss.com/rss/6774461d2fe478aa030177026774460744e7b82c5407c544.rss'
    
    try:
        all_news_articles = []
        
        # Fetch and parse the RSS feed
        feed = fetch_rss_feed(rss_url)
        
        for entry in feed.entries:
            image_url = entry.media_content[0]['url'] if hasattr(entry, "media_content") and entry.media_content else None
            summary = entry.description.strip() if entry.description else None
            all_news_articles.append({
                'title': entry.title,
                'link': entry.link,
                'summary': summary,
                'published': entry.published,
                'image_url': image_url
            })
        
        # Return the list of news articles from the RSS feed
        return jsonify(all_news_articles)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return "Welcome to the Live App!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
