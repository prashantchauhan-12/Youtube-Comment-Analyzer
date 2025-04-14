import re
import os
import pandas as pd
import threading
import matplotlib.pyplot as plt
from textblob import TextBlob
from flask import Flask, render_template, request, jsonify, send_file
from googleapiclient.discovery import build
from collections import Counter

# Get API key from environment variable for security
YOUTUBE_API_KEY = "AIzaSyCO3p461iaNZbGECphVjIORTUNpRb7TOSs"  # Temporary solution
app = Flask(__name__)

# Create directories if they don't exist
os.makedirs("static/charts", exist_ok=True)

# Store analysis results for quick retrieval
analysis_cache = {}
processing_videos = set()

def get_video_id(url):
    """Extracts video ID from a YouTube URL."""
    # Support various YouTube URL formats
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and shortened URLs
        r'(?:embed\/)([0-9A-Za-z_-]{11})',   # Embed URLs
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'  # youtu.be URLs
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_comments(video_id, max_results=500):
    """Fetches comments from a YouTube video with error handling."""
    if not YOUTUBE_API_KEY:
        return {"error": "YouTube API key not configured. Please set the YOUTUBE_API_KEY environment variable."}
        
    youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
    comments = []
    
    try:
        # Get video info first to check if comments are enabled
        video_response = youtube.videos().list(
            part="statistics,snippet",
            id=video_id
        ).execute()
        
        if not video_response.get("items"):
            return {"error": "Video not found"}
        
        video_info = video_response["items"][0]
        video_title = video_info["snippet"]["title"]
        comment_count = int(video_info["statistics"].get("commentCount", 0))
        
        if comment_count == 0:
            return {"error": "Comments are disabled for this video", "title": video_title}
        
        # Now get comments
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            textFormat="plainText"
        )
        response = request.execute()
        
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "text": comment["textDisplay"],
                "author": comment["authorDisplayName"],
                "date": comment["publishedAt"],
                "likes": comment["likeCount"]
            })
        
        return {
            "comments": comments, 
            "title": video_title,
            "comment_count": comment_count
        }
    
    except Exception as e:
        if "quota" in str(e).lower():
            return {"error": "YouTube API quota exceeded. Please try again later."}
        return {"error": f"An error occurred: {str(e)}"}

def generate_crater_summary(results_df, summary_stats):
    """Generate a concise crater summary of the sentiment analysis results"""
    # Determine overall sentiment trend
    if summary_stats["positive_percentage"] > 60:
        overall_trend = "Overwhelmingly positive"
    elif summary_stats["positive_percentage"] > 40:
        overall_trend = "Generally positive"
    elif summary_stats["negative_percentage"] > 60:
        overall_trend = "Overwhelmingly negative"
    elif summary_stats["negative_percentage"] > 40:
        overall_trend = "Generally negative"
    else:
        overall_trend = "Mixed or neutral"
    
    # Identify most liked comments by sentiment category
    top_positive = results_df[results_df["sentiment"] == "Positive"].nlargest(1, "likes")
    top_negative = results_df[results_df["sentiment"] == "Negative"].nlargest(1, "likes")
    
    # Find extreme sentiment comments
    most_positive = results_df.nlargest(1, "polarity")
    most_negative = results_df.nsmallest(1, "polarity")
    
    # Prepare crater summary
    crater_summary = {
        "overall_trend": overall_trend,
        "sentiment_ratio": f"{summary_stats['positive_percentage']}% positive, {summary_stats['neutral_percentage']}% neutral, {summary_stats['negative_percentage']}% negative",
        "top_positive_comment": top_positive["comment"].iloc[0] if not top_positive.empty else "No positive comments found",
        "top_negative_comment": top_negative["comment"].iloc[0] if not top_negative.empty else "No negative comments found",
        "strongest_positive": most_positive["comment"].iloc[0] if not most_positive.empty else "No comments found",
        "strongest_negative": most_negative["comment"].iloc[0] if not most_negative.empty else "No comments found",
        "average_sentiment_score": summary_stats["average_polarity"]
    }
    
    return crater_summary

# Implementation of missing functions for the creator summary
def get_sentiment_overview(summary_stats):
    """Generate sentiment overview based on summary stats"""
    if summary_stats["positive_percentage"] > 70:
        return "Overwhelmingly positive reception"
    elif summary_stats["positive_percentage"] > 50:
        return "Generally positive reception"
    elif summary_stats["negative_percentage"] > 50:
        return "Generally negative reception"
    elif summary_stats["negative_percentage"] > 70:
        return "Overwhelmingly negative reception"
    else:
        return "Mixed reception with balanced opinions"

def get_engagement_level(df):
    """Determine engagement level based on likes and comment length"""
    avg_likes = df["likes"].mean() if not df.empty else 0
    avg_length = df["comment"].apply(len).mean() if not df.empty else 0
    
    if avg_likes > 10 and avg_length > 100:
        return "High engagement with detailed feedback"
    elif avg_likes > 5 or avg_length > 50:
        return "Moderate engagement"
    else:
        return "Low engagement"

def get_audience_alignment(df):
    """Determine audience alignment based on sentiment consistency"""
    sentiment_counts = Counter(df["sentiment"])
    total = len(df)
    if total == 0:
        return "Insufficient data to determine audience alignment"
    
    max_percentage = max([
        sentiment_counts.get("Positive", 0) / total,
        sentiment_counts.get("Negative", 0) / total,
        sentiment_counts.get("Neutral", 0) / total
    ]) * 100
    
    if max_percentage > 70:
        return "Strong audience consensus"
    elif max_percentage > 50:
        return "Moderate audience consensus"
    else:
        return "Divided audience opinion"

def analyze_content_feedback(df):
    """Analyze feedback related to video content"""
    content_keywords = ["content", "topic", "subject", "information", "accurate", "helpful", "useful"]
    content_comments = df[df["comment"].str.lower().apply(lambda x: any(word in x for word in content_keywords))]
    
    positive_content = content_comments[content_comments["sentiment"] == "Positive"].shape[0]
    total_content = content_comments.shape[0]
    
    if total_content == 0:
        return "No specific content feedback detected"
    
    positive_ratio = positive_content / total_content if total_content > 0 else 0
    
    if positive_ratio > 0.7:
        return "Content is very well-received"
    elif positive_ratio > 0.5:
        return "Content is generally well-received"
    else:
        return "Mixed feedback on content"

def analyze_presentation_feedback(df):
    """Analyze feedback related to presentation style"""
    presentation_keywords = ["explain", "presentation", "voice", "clear", "understand", "pace", "speed"]
    presentation_comments = df[df["comment"].str.lower().apply(lambda x: any(word in x for word in presentation_keywords))]
    
    positive_presentation = presentation_comments[presentation_comments["sentiment"] == "Positive"].shape[0]
    total_presentation = presentation_comments.shape[0]
    
    if total_presentation == 0:
        return "No specific presentation feedback detected"
    
    positive_ratio = positive_presentation / total_presentation if total_presentation > 0 else 0
    
    if positive_ratio > 0.7:
        return "Presentation style is very well-received"
    elif positive_ratio > 0.5:
        return "Presentation style is generally well-received"
    else:
        return "Mixed feedback on presentation style"

def analyze_technical_feedback(df):
    """Analyze feedback related to technical aspects"""
    technical_keywords = ["quality", "audio", "video", "sound", "resolution", "edit", "editing"]
    technical_comments = df[df["comment"].str.lower().apply(lambda x: any(word in x for word in technical_keywords))]
    
    positive_technical = technical_comments[technical_comments["sentiment"] == "Positive"].shape[0]
    total_technical = technical_comments.shape[0]
    
    if total_technical == 0:
        return "No specific technical feedback detected"
    
    positive_ratio = positive_technical / total_technical if total_technical > 0 else 0
    
    if positive_ratio > 0.7:
        return "Technical aspects are very well-received"
    elif positive_ratio > 0.5:
        return "Technical aspects are generally well-received"
    else:
        return "Mixed feedback on technical aspects"

def analyze_engagement_indicators(df):
    """Analyze indicators of audience engagement"""
    engagement_keywords = ["more", "please", "next", "love", "subscribe", "fan", "follow"]
    engagement_comments = df[df["comment"].str.lower().apply(lambda x: any(word in x for word in engagement_keywords))]
    
    positive_engagement = engagement_comments[engagement_comments["sentiment"] == "Positive"].shape[0]
    total_engagement = engagement_comments.shape[0]
    
    if total_engagement == 0:
        return "Limited engagement indicators detected"
    
    positive_ratio = positive_engagement / total_engagement if total_engagement > 0 else 0
    
    if positive_ratio > 0.7:
        return "Strong positive engagement indicators"
    elif positive_ratio > 0.5:
        return "Moderate positive engagement indicators"
    else:
        return "Mixed engagement indicators"

def generate_improvement_suggestions(negative_comments):
    """Generate improvement suggestions based on negative comments"""
    if len(negative_comments) == 0:
        return ["No specific improvement suggestions detected"]
    
    # Simple keyword-based suggestion generation
    improvements = []
    
    content_keywords = ["confusing", "confused", "unclear", "wrong", "incorrect", "mistake"]
    if any(any(word in comment.lower() for word in content_keywords) for comment in negative_comments):
        improvements.append("Consider clarifying content points that viewers found confusing")
    
    presentation_keywords = ["boring", "slow", "fast", "hard to follow", "monotone"]
    if any(any(word in comment.lower() for word in presentation_keywords) for comment in negative_comments):
        improvements.append("Review presentation style based on viewer feedback")
    
    technical_keywords = ["audio", "sound", "quality", "resolution", "editing"]
    if any(any(word in comment.lower() for word in technical_keywords) for comment in negative_comments):
        improvements.append("Address technical issues mentioned in comments")
    
    if not improvements:
        improvements.append("Review negative comments for specific improvement areas")
    
    return improvements

def generate_viewer_praise(positive_comments):
    """Extract key praise points from positive comments"""
    if len(positive_comments) == 0:
        return ["No specific praise detected"]
    
    # Simple keyword-based praise extraction
    praise = []
    
    content_keywords = ["informative", "helpful", "useful", "great info", "learned"]
    if any(any(word in comment.lower() for word in content_keywords) for comment in positive_comments):
        praise.append("Content is appreciated for being informative and helpful")
    
    presentation_keywords = ["clear", "well explained", "engaging", "entertaining"]
    if any(any(word in comment.lower() for word in presentation_keywords) for comment in positive_comments):
        praise.append("Presentation style is well-received")
    
    quality_keywords = ["quality", "professional", "well made", "excellent"]
    if any(any(word in comment.lower() for word in quality_keywords) for comment in positive_comments):
        praise.append("Overall quality of content is appreciated")
    
    if not praise:
        praise.append("General positive sentiment detected")
    
    return praise

def generate_creator_summary(comments_df, summary_stats):
    """Generate a detailed summary for content creators"""
    positive_comments = comments_df[comments_df['sentiment'] == 'Positive']
    negative_comments = comments_df[comments_df['sentiment'] == 'Negative']
    
    # Common positive feedback themes
    positive_themes = analyze_common_themes(positive_comments['comment'].tolist() if not positive_comments.empty else [])
    
    # Common negative feedback themes
    negative_themes = analyze_common_themes(negative_comments['comment'].tolist() if not negative_comments.empty else [])
    
    # Common questions/requests
    questions = analyze_questions(comments_df['comment'].tolist() if not comments_df.empty else [])
    
    # Improvement suggestions
    suggestions = generate_improvement_suggestions(negative_comments['comment'].tolist() if not negative_comments.empty else [])
    
    # Viewer praise
    praise = generate_viewer_praise(positive_comments['comment'].tolist() if not positive_comments.empty else [])
    
    return {
        "sentiment_overview": get_sentiment_overview(summary_stats),
        "engagement_level": get_engagement_level(comments_df),
        "audience_alignment": get_audience_alignment(comments_df),
        "positive_feedback": positive_themes,
        "negative_feedback": negative_themes,
        "questions_requests": questions,
        "content_feedback": analyze_content_feedback(comments_df),
        "presentation_feedback": analyze_presentation_feedback(comments_df),
        "technical_feedback": analyze_technical_feedback(comments_df),
        "engagement_indicators": analyze_engagement_indicators(comments_df),
        "improvement_suggestions": suggestions,
        "viewer_praise": praise
    }

def analyze_common_themes(comments, n=5):
    """Identify common themes in comments"""
    # Implementation would use NLP techniques to cluster similar comments
    # This is a simplified version
    if not comments:
        return [{"topic": "No data available", "count": 0}]
        
    all_text = ' '.join(comments).lower()
    words = re.findall(r'\b[a-z]{3,15}\b', all_text)
    stop_words = ['the', 'and', 'to', 'of', 'is', 'in', 'it', 'that', 'this', 'was']
    filtered_words = [word for word in words if word not in stop_words]
    
    if not filtered_words:
        return [{"topic": "No significant keywords found", "count": 0}]
        
    word_counts = Counter(filtered_words).most_common(n)
    
    return [{"topic": word, "count": count} for word, count in word_counts]

def analyze_questions(comments):
    """Identify common questions in comments"""
    # Simplified implementation - would use NLP in production
    questions = []
    for comment in comments:
        if '?' in comment:
            questions.append(comment)
    
    # Group similar questions
    return [{"topic": "Common questions", "count": len(questions)}]

def analyze_sentiment(comments_data):
    """Analyzes sentiment of comments with more detailed metrics."""
    if "error" in comments_data:
        return comments_data
    
    comments = comments_data["comments"]
    results = []
    
    for comment in comments:
        text = comment["text"]
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Categorize sentiment
        if polarity > 0.2:
            label = "Positive"
        elif polarity < -0.2:
            label = "Negative"
        else:
            label = "Neutral"
        
        results.append({
            "comment": text,
            "author": comment["author"],
            "date": comment["date"],
            "likes": comment["likes"],
            "sentiment": label,
            "polarity": round(polarity, 2),
            "subjectivity": round(subjectivity, 2)
        })
    
    df = pd.DataFrame(results)
    
    # Generate summary statistics
    sentiment_counts = Counter(df["sentiment"])
    total = len(df)
    summary = {
        "positive_percentage": round((sentiment_counts.get("Positive", 0) / total) * 100, 1) if total > 0 else 0,
        "negative_percentage": round((sentiment_counts.get("Negative", 0) / total) * 100, 1) if total > 0 else 0,
        "neutral_percentage": round((sentiment_counts.get("Neutral", 0) / total) * 100, 1) if total > 0 else 0,
        "average_polarity": round(df["polarity"].mean(), 2) if not df.empty else 0,
        "average_subjectivity": round(df["subjectivity"].mean(), 2) if not df.empty else 0,
    }
    
    # Generate crater summary
    crater_summary = generate_crater_summary(df, summary)
    
    # Add common words analysis
    all_text = " ".join(df["comment"].tolist()).lower()
    words = re.findall(r'\b[a-z]{3,15}\b', all_text)
    # Remove common stop words
    stop_words = ['the', 'and', 'to', 'of', 'is', 'in', 'it', 'that', 'this', 'was', 'for', 'on', 'with', 
                 'you', 'are', 'have', 'be', 'at', 'not', 'but', 'they', 'from', 'he', 'she', 'his', 'her', 
                 'their', 'as', 'by', 'an', 'so', 'if', 'or', 'just', 'can', 'would', 'could', 'should', 
                 'what', 'how', 'when', 'where', 'why', 'who', 'all', 'any', 'very', 'your', 'there', 'will']
    filtered_words = [word for word in words if word not in stop_words]
    word_freq = Counter(filtered_words).most_common(20)
    common_words = [{"word": word, "count": count} for word, count in word_freq]
    
    # Generate charts
    generate_charts(df, sentiment_counts, comments_data["title"])
    
    # Save results to CSV file
    df.to_csv("static/sentiment_analysis_results.csv", index=False)
    
    # Create complete result package
    results = {
        "results": df.to_dict('records'),
        "summary": summary,
        "common_words": common_words,
        "crater_summary": crater_summary,
        "title": comments_data["title"],
        "total_comments": comments_data["comment_count"],
        "analyzed_comments": len(df),
        "creator_summary": generate_creator_summary(df, summary)
    }
    
    return results

def generate_charts(df, sentiment_counts, title):
    """Generate visualization charts for the analysis"""
    # Create static/charts directory if it doesn't exist
    if not os.path.exists("static/charts"):
        os.makedirs("static/charts")
    
    # 1. Sentiment distribution chart
    plt.figure(figsize=(8, 5))
    colors = ['#4CAF50', '#FFC107', '#F44336']  # green, yellow, red
    labels = ['Positive', 'Neutral', 'Negative']
    values = [sentiment_counts.get(label, 0) for label in labels]
    
    plt.bar(labels, values, color=colors)
    plt.title(f'Sentiment Distribution for "{title[:30]}..."' if len(title) > 30 else f'Sentiment Distribution for "{title}"')
    plt.ylabel('Number of Comments')
    plt.tight_layout()
    plt.savefig('static/charts/sentiment_distribution.png')
    plt.close()
    
    # 2. Polarity distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df['polarity'], bins=20, color='#2196F3', alpha=0.7)
    plt.title('Comment Polarity Distribution')
    plt.xlabel('Polarity (-1: Very Negative, 1: Very Positive)')
    plt.ylabel('Number of Comments')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('static/charts/polarity_distribution.png')
    plt.close()
    
    # 3. New chart for crater summary visualization
    plt.figure(figsize=(8, 5))
    sentiment_values = [sentiment_counts.get('Positive', 0), 
                        sentiment_counts.get('Neutral', 0), 
                        sentiment_counts.get('Negative', 0)]
    sentiment_labels = ['Positive', 'Neutral', 'Negative']
    plt.pie(sentiment_values, labels=sentiment_labels, autopct='%1.1f%%', 
            colors=['#4CAF50', '#FFC107', '#F44336'], startangle=90)
    plt.axis('equal')
    plt.title('Sentiment Distribution Pie Chart')
    plt.tight_layout()
    plt.savefig('static/charts/sentiment_pie.png')
    plt.close()

def analyze_video_comments(video_id, max_comments):
    """Run sentiment analysis and cache results"""
    try:
        processing_videos.add(video_id)  # Mark as processing
        comments_data = get_comments(video_id, max_comments)
        if "error" not in comments_data:
            analysis_results = analyze_sentiment(comments_data)
            analysis_cache[video_id] = analysis_results
        else:
            analysis_cache[video_id] = comments_data
    finally:
        # Make sure we remove from processing set even if there's an error
        if video_id in processing_videos:
            processing_videos.remove(video_id)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    url = request.form.get('url')
    max_comments = int(request.form.get('max_comments', 100))
    
    video_id = get_video_id(url)
    if not video_id:
        return jsonify({"error": "Invalid YouTube URL. Please provide a valid YouTube video link."})
    
    # Check if analysis is already in progress
    if video_id in processing_videos:
        return jsonify({"status": "Analysis already in progress", "video_id": video_id})
    
    # Check if we already have results
    if video_id in analysis_cache:
        return jsonify({"status": "Analysis complete", "video_id": video_id})
    
    # Run analysis in background thread to not block the response
    threading.Thread(target=analyze_video_comments, args=(video_id, max_comments)).start()
    
    return jsonify({"status": "Analysis started", "video_id": video_id})

@app.route('/results/<video_id>')
def get_results(video_id):
    # Check if analysis is complete
    if video_id in analysis_cache:
        return jsonify(analysis_cache[video_id])
    
    # Check if analysis is in progress
    if video_id in processing_videos:
        return jsonify({"status": "processing"})
    
    # If neither, start a new analysis with default parameters
    threading.Thread(target=analyze_video_comments, args=(video_id, 100)).start()
    return jsonify({"status": "Analysis started", "video_id": video_id})

@app.route('/download')
def download_results():
    if os.path.exists("static/sentiment_analysis_results.csv"):
        return send_file('static/sentiment_analysis_results.csv',
                        as_attachment=True,
                        download_name='sentiment_analysis_results.csv')
    else:
        return jsonify({"error": "No analysis results available for download."})

if __name__ == "__main__":
    # Check if API key is set
    if not YOUTUBE_API_KEY:
        print("WARNING: YouTube API key not set. Please set the YOUTUBE_API_KEY environment variable.")
        print("You can set it by running: export YOUTUBE_API_KEY='your_api_key'")
    
    app.run(debug=True)