from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import requests
import re
# import whisper
from rest_framework.decorators import api_view
from rest_framework.response import Response

# Domain-specific criteria (can be loaded dynamically from a config or database)
DOMAIN_CRITERIA = {
    "positive_keywords": ['satisfied', 'happy', 'good service', 'interested', 'details', 'budget'],
    "negative_keywords": ['unhappy', 'bad service', 'not interested', 'busy', 'poor', 'expensive'],
    "project_names": ['mall of imarat', 'bavylon', 'imarat cyber tower', 'golf floras'],
    "project_types": ['commercial', 'residential'],
    "product_types": ['installment', 'rental']
}

# Utility functions
def calculate_sentiment_score(sentiments):
    """
    Calculate sentiment score from Whisper sentiment analysis data.
    """
    score = 0
    for segment in sentiments:
        if segment['sentiment'] == 'positive':
            score += segment['sentiment_score'] * 10
        elif segment['sentiment'] == 'negative':
            score -= abs(segment['sentiment_score']) * 10
    return max(0, score)

def analyze_keywords(transcript, keywords):
    """
    Analyze transcript for the presence of keywords and calculate a score.
    """
    score = 0
    for keyword in keywords:
        if re.search(rf'\b{keyword}\b', transcript, re.IGNORECASE):
            score += 10  # Assign score per keyword found
    return score

def calculate_scores(whisper_data):
    """
    Analyze Whisper JSON response and calculate dynamic scores.
    """
    transcript = whisper_data["results"]["channels"][0]["alternatives"][0]["paragraphs"]["transcript"]
    sentiments = whisper_data["results"]["sentiments"]["segments"]

    # Calculate sentiment score
    sentiment_score = calculate_sentiment_score(sentiments)

    # Calculate keyword-based scores
    positive_score = analyze_keywords(transcript, DOMAIN_CRITERIA["positive_keywords"])
    negative_score = analyze_keywords(transcript, DOMAIN_CRITERIA["negative_keywords"])
    project_score = analyze_keywords(transcript, DOMAIN_CRITERIA["project_names"])

    # Dynamic scoring logic
    total_score = sentiment_score + positive_score - negative_score + project_score
    return {
        "sentiment_score": sentiment_score,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "project_score": project_score,
        "total_score": total_score
    }

# @api_view(['POST'])
@csrf_exempt
def whisper_analysis_api(request):
    """
    Django view to analyze audio files using Whisper AI.
    """
    # if request.method == "POST":
    try:
        # Parse incoming JSON request
        # body = json.loads(request.body)
        # file_url = body.get("file_url")  # For external file handling (future use)
        file_path = "data/out-03329037737-1006-20240708-164521-1720439121.23153.wav"
        return file_path
        # Validate input
        if not file_path:
            return JsonResponse({"error": "File path not provided"}, status=400)

        # # Load Whisper model
        # model = whisper.load_model("large")

        # # Transcribe the audio file
        # result = model.transcribe(file_path, fp16=False, task="translate")
        # print('\n--start--\n',result,'\n--end--')
        # transcript = result.get('text', '')
        
        # # if transcript_only:
        # return JsonResponse({"transcript": transcript}, status=200)

        # Process Whisper's result for analysis

        scores = calculate_scores(result)
        return JsonResponse({
            "status": "success",
            "scores": scores,
            "details": result  # Optionally include full Whisper data
        }, status=200)

    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": str(e)
        }, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)
