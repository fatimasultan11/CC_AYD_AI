from django.http import HttpResponse
from django.http import JsonResponse
import os
import re
import json
import requests
#import matplotlib.pyplot as plt
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
import plotly.express as px
import pandas as pd
import whisper

# from deepgram import (
#     DeepgramClient,
#     PrerecordedOptions,
#     DeepgramClientOptions,
#     FileSource,
# )
import httpx
load_dotenv()

# Path to the audio file
AUDIO_FILE = "data/downloaded.wav"

API_KEY = "3f4d292c6ced23ca5bbf82ce2e76091b16d8dbaa"
def sort_by_sentiment(sentiment_dict):
  if sentiment_dict["sentiment"] == "positive":
    return 1
  else:
    return 0

def calculate_sentiment_score(score: float) -> int:
    if score >= 0.05:
        return score * 200  # Positive sentiment
    elif score <= -0.05:
        return 0  # Negative sentiment
    else:
        return 50  # Neutral sentiment

def calculate_keyword_score(transcript: str, keywords: list) -> int:
    score = 0
    for keyword in keywords:
        if re.search(r'\b' + keyword + r'\b', transcript, re.IGNORECASE):
            score += 10  # Assign 2 points for each keyword found
    return score

def calculate_duration_score(duration: int) -> int:
    if duration < 2:
        return 0  # Too short
    elif duration > 10:
        return 0  # Too long
    else:
        return 5  # Optimal duration

def calculate_satisfaction_score(transcript: str) -> int:
    positive_indicators = ['satisfied', 'happy', 'good service', 'thank', 'thanks', 'nice', 'best','interested', 'detail', 'details','good', 'need information']
    negative_indicators = ['unhappy', 'bad service', 'not satisfied', 'nothing happened', 'number', 'personal','not interested', 'busy', 'not answering', 'bad','poor', 'expensive', 'no budget']
    
    score = 0
    for phrase in positive_indicators:
        if re.search(r'\b' + phrase + r'\b', transcript, re.IGNORECASE):
            score += 10
    
    for phrase in negative_indicators:
        if re.search(r'\b' + phrase + r'\b', transcript, re.IGNORECASE):
            score -= 10
    
    return score

def calculate_call_score(transcript: str, duration: int, sentiment_value) -> int:
    keywords = ['emirates', 'imarat', 'imaraat', 'graana','agency21','investment', 'installment', 'budget','real estate', 'commercial','project', 'detail', 'share', 'information', 'tower', 'apartment', 'shop', 'residential','buy','rent','sell','house','building','construction','company', 'sold']
    
    sentiment_score = calculate_sentiment_score(sentiment_value)
    keyword_score = calculate_keyword_score(transcript, keywords)
    duration_score = calculate_duration_score(duration)
    satisfaction_score = calculate_satisfaction_score(transcript)
    
    total_score = int((25*sentiment_score + 25*keyword_score + 25*duration_score + 25*satisfaction_score)/100)
    return total_score

def download_audio_file(url: str):
    try:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        filename = "data/" + query_params.get('filename', ['downloaded_audio.wav'])[0]
        
        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful
        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192): 
                file.write(chunk)
        print(f"File downloaded and saved as {filename}")
        return filename
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

def plot_sentiments(sentiment_scores):
    df = pd.DataFrame(sentiment_scores, columns=['Sentiment Score', 'Start Time', 'End Time'])
    #df = sentiment_scores
    fig = px.line(df, x='Start Time', y='Sentiment Score', title='Sentiment Score Over Time', markers=True)
    fig.show()
    # plt.figure(figsize=(10, 5))
    # plt.plot(sentiment_scores, marker='o', linestyle='-', color='b')
    # plt.title('Sentiment Scores')
    # plt.xlabel('Paragraph Index')
    # plt.ylabel('Sentiment Score')
    # plt.grid(True)
    # plt.show()

def calculate_greetings_score(transcript: str):
    keywords = ['alaikum', 'how are you', 'my name', 'speaking from', 'calling from' ]
    score = 0
    for keyword in keywords:
        if re.search(r'\b' + keyword + r'\b', transcript, re.IGNORECASE):
            score += 1
    if score > 2: return 20
    if score == 2: return 15
    if score == 1: return 10
    return 0
def calculate_company_intro_score(transcript: str):
    keywords = ['real estate company', 'construction company', 'real estate developer', 'real estate builder' ]
    for keyword in keywords:
        if re.search(r'\b' + keyword + r'\b', transcript, re.IGNORECASE):
            return 20
    return 0
def calculate_kyc_score(transcript: str):
    keywords = ['your name', 'city', 'may i know', 'what is your' ]
    score = 0
    for keyword in keywords:
        if re.search(r'\b' + keyword + r'\b', transcript, re.IGNORECASE):
            score += 5
    return score
def calculate_pitching_score(transcript: str):
    keywords = ['project', 'investment', 'commercial', 'residential','shop', 'budget','buy','sell','rent', 'plan' ]
    score = 0
    for keyword in keywords:
        if re.search(r'\b' + keyword + r'\b', transcript, re.IGNORECASE):
            score += 1
    if score > 3: return 20
    if score == 3: return 15
    if score == 2: return 10
    return 0
def calculate_ending_score(transcript: str):
    keywords = ['allah hafiz', 'bye', 'take care']
    score = 0
    for keyword in keywords:
        if re.search(r'\b' + keyword + r'\b', transcript, re.IGNORECASE):
            return 20
    return 0
def calculate_agent_score(transcript: str, result):
    result['agent_score'] = {}
    result['agent_score']['greetings_score'] = calculate_greetings_score(transcript)
    result['agent_score']['company_intro_score'] = calculate_company_intro_score(transcript)
    result['agent_score']['kyc_score'] = calculate_kyc_score(transcript)
    result['agent_score']['pitching_score'] = calculate_pitching_score(transcript)
    result['agent_score']['call_ending_score'] = calculate_ending_score(transcript)
    result['agent_score']['agent_score'] = result['agent_score']['greetings_score'] + result['agent_score']['company_intro_score'] + result['agent_score']['kyc_score'] + result['agent_score']['pitching_score'] + result['agent_score']['call_ending_score']

def get_project_info(transcript: str, result):
    result['pitching_details'] = {'project_name':'', 'project_type_commercial': False, 'project_type_residential': False, 'product_type_installment': False, 'product_type_rental': False}
    project_names = ['mall of imarat', 'bavylon', 'imarat cyber tower', 'grand bazar', 'imarat builders mall', 'imarat residences', 'amazon mall', 'golf floras','florence galleria', 'hoon adventure park']
    project_type_commercial = ['commercial', 'shop', 'mall']
    project_type_residential = ['residential', 'flat', 'apartment']
    product_type_installment = ['installment', 'installments', 'down payment', 'quarterly', 'monthly']
    product_type_rental = ['rental', 'rent', 'appreciation']
    for keyword in project_names:
        if re.search(r'\b' + keyword + r'\b', transcript, re.IGNORECASE):
            result['pitching_details']['project_name'] = keyword
    for keyword in project_type_commercial:
        if re.search(r'\b' + keyword + r'\b', transcript, re.IGNORECASE):
            result['pitching_details']['project_type_commercial'] = True
    for keyword in project_type_residential:
        if re.search(r'\b' + keyword + r'\b', transcript, re.IGNORECASE):
            result['pitching_details']['project_type_residential'] = True
    for keyword in product_type_installment:
        if re.search(r'\b' + keyword + r'\b', transcript, re.IGNORECASE):
            result['pitching_details']['product_type_installment'] = True
    for keyword in product_type_rental:
        if re.search(r'\b' + keyword + r'\b', transcript, re.IGNORECASE):
            result['pitching_details']['product_type_rental'] = True
    return transcript
def get_entities(data, result):
    if data["results"]["channels"][0]['alternatives'][0]['entities']:
        entities = data["results"]["channels"][0]['alternatives'][0]['entities']
        filtered_entities = [entity for entity in entities if entity["confidence"] > 0.9]

        # Step 2: Remove duplicates based on label
        unique_entities = {}
        for entity in filtered_entities:
            if entity["label"] not in unique_entities:
                unique_entities[entity["label"]] = entity

        # Step 3: Remove values containing 'alaikum'
        final_entities = [
            entity for entity in unique_entities.values() if "alaikum" not in entity["value"].lower()
        ]
        valid_labels = {'NAME', 'LOCATION', 'EMAIL', 'PHONE_NUM', 'CREDIT_CARD', 'CARD_NUM'}
        filtered_entities = [entity for entity in final_entities if entity['label'] in valid_labels]

        # Step 2: Replace 'CREDIT_CARD' and 'CARD_NUM' labels with 'PHONE_NUM'
        for entity in filtered_entities:
            if entity['label'] in {'CREDIT_CARD', 'CARD_NUM'}:
                entity['label'] = 'PHONE_NUM'
        result['entities'] = filtered_entities
    else:
        result['entities'] = []

def get_response(filename):
    with open(filename, 'r', encoding='utf-8') as f:
    # Load the JSON data from the file
        data = json.load(f)
        data = json.loads(data)
    result = {}
    result["transcript"] = data["results"]["channels"][0]['alternatives'][0]['paragraphs']['transcript']
    result["transcript"] = re.sub(r'emirates', 'imarat', result["transcript"], flags=re.IGNORECASE)
    result["transcript"] = re.sub(r'imaraat', 'imarat', result["transcript"], flags=re.IGNORECASE)
    result["transcript"] = re.sub(r'bazaar', 'bazar', result["transcript"], flags=re.IGNORECASE)
    result["summary"] = data["results"]["channels"][0]['alternatives'][0]['summaries'][0]['summary']
    result["overall_sentiment"] = data["results"]["sentiments"]["average"]["sentiment"]
    result["overal_sentiment_score"] = data["results"]["sentiments"]["average"]["sentiment_score"]
    result['sentiment_segments'] = data["results"]["sentiments"]["segments"]
    result['sentiment_segments'].sort(key=sort_by_sentiment, reverse=True) 

    segments = data["results"]["topics"]["segments"]
    all_topics = []
    for segment in segments:
        topics = segment["topics"]
        for topic_dict in topics:
            all_topics.append(topic_dict['topic'])
    result["topics"] = all_topics
    
    segments = data["results"]["intents"]["segments"]
    all_intents = []
    for segment in segments:
        intents = segment["intents"]
        for intent_dict in intents:
            all_intents.append(intent_dict['intent'])
    result["intents"] = all_intents

    #paragraphs = data["results"]["channels"][0]['alternatives'][0]['paragraphs']['paragraphs']
    paragraphs = data["results"]["sentiments"]["segments"]
    filtered_paragraphs = paragraphs#[para for para in paragraphs if para['speaker'] == 0]
    #filtered_paragraphs = [para for para in filtered_paragraphs if para['sentiment'] != 'neutral']
    negative_paragraphs = [para for para in paragraphs if para['sentiment'] == 'negative']

    paragraphs = data["results"]["channels"][0]['alternatives'][0]['paragraphs']['paragraphs']
    sentiment_scores = [(round(paragraph['sentiment_score'], 2), paragraph['start'], paragraph['end']) for paragraph in paragraphs]
    print(sentiment_scores)
    result['sentiment_scores'] = sentiment_scores
    plot_sentiments(sentiment_scores=sentiment_scores)
    sentiment_sums = {
    "neutral": 0.0,
    "positive": 0.0,
    "negative": 0.0
    }
    sentiment_counts = {
        "neutral": 0,
        "positive": 0,
        "negative": 0
    }

    # Sum sentiment scores and count sentiments based on sentiment
    for para in filtered_paragraphs:
        sentiment = para['sentiment']
        sentiment_sums[sentiment] += para['sentiment_score']
        sentiment_counts[sentiment] += 1
    average_sentiment_scores = {
        sentiment: (sentiment_sums[sentiment] / sentiment_counts[sentiment] if sentiment_counts[sentiment] > 0 else 0)
        for sentiment in sentiment_sums
    }
    total_sum = sum(sentiment_sums.values())
    total_count = sum(sentiment_counts.values())
    CSAT = int((((total_sum / total_count) + 1)*100)/2) if total_count > 0 else 0
    result["sentiment_counts"] = sentiment_counts
    result["sentiment_sums"] = sentiment_sums
    result["average_sentiment_scores"] = average_sentiment_scores
    result['negative_paragraphs'] = negative_paragraphs
    result["CSAT"] = CSAT

    result['call_score'] = calculate_call_score(transcript=result["transcript"], duration=data['metadata']['duration'],sentiment_value=result['overal_sentiment_score'])
    calculate_agent_score(transcript=result['transcript'], result=result)
    get_project_info(transcript=result['transcript'], result=result)
    #get_entities(data=data, result=result)
    result['entities'] = []
    #return JsonResponse(data)
    #return JsonResponse(result)
    return result
# def index22(request):
#     print('request\n',request)
#     #result = get_response(filename="data/out-03329037737-1006-20240708-164521-1720439121.23153.wav.json")
#     #return JsonResponse(result)
#     # url = request.GET.get('url', '')   
#     # print('\nurl\n',url) 
#     # filename = download_audio_file(url=url)
#     filename = "data/out-03329037737-1006-20240708-164521-1720439121.23153.wav"
#     # filename = "data/in-4238020121-03153721111-20240627-114232-1719470552.16797.wav"
#     try:
#         deepgram = DeepgramClient(api_key=API_KEY)
#         with open(filename, "rb") as file:
#             buffer_data = file.read()

#         payload: FileSource = {
#             "buffer": buffer_data,
#         }

#         #STEP 2: Configure Deepgram options for audio analysis
#         options = PrerecordedOptions(
#             #model="nova-2",
#             # model="whisper-large",
#             model="whisper-large",
#             smart_format=True,
#             detect_topics=True,
#             #detect_entities=True,
#             diarize=True,
#             intents=True,
#             sentiment=True,
#             punctuate=False,
#             summarize=True,
#             topics=True,
#             #language="hi"
#         )

#         # STEP 3: Call the transcribe_file method with the text payload and options
#         response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options, timeout=httpx.Timeout(timeout=None))
#         #response = deepgram.listen.prerecorded.v("1").transcribe_url(AUDIO_URL, options)

#         with open(filename + '.json', 'w', encoding='utf-8') as f:
#             json.dump(response.to_json(), f, ensure_ascii=False)
#         #response = get_response(response)
#         # STEP 4: Print the response
#         #print(response)
#         #return HttpResponse(response)
#         result = get_response(filename=filename + '.json')
#         result['call_id'] = request.GET.get('call_id', '')
#         result['agent_id'] = request.GET.get('agent_id', '')
#         result['call_direction'] = request.GET.get('call_direction', '')
#         return JsonResponse(result)
#     except Exception as e:
#         print(f"Exception: {e}")
#         return HttpResponse(e)
#     #return HttpResponse("Hello, world. You're at the rabtAI index.")
    
import whisper
from django.http import HttpResponse
import os

def index(request):
    # Assuming the file is uploaded via a request or exists in the "data" directory
    # filename = "data/out-03329037737-1006-20240708-164521-1720439121.23153.wav"
    filename = "data/sample.wav"
    
    if not os.path.exists(filename):
        return HttpResponse("File not found", status=404)
    
    try:
        model = whisper.load_model("large")

        # Transcribe the audio file
        result = model.transcribe(filename,fp16=False, task="translate")
        print('\n--start--\n',result,'\n--end--')
        # Return transcription result
        return HttpResponse(result['text'])

    except Exception as e:
        # Handle exceptions and return the error in the response
        return HttpResponse(f"Exception occurred--: {e,'--',os.path.exists(filename)}", status=500,)

#  file def index(request): 
def file_index(request):
    if request.method == 'POST' and request.FILES['audio_file']:
        audio_file = request.FILES['audio_file']
        filename = os.path.join('data', audio_file.name)
        with open(filename, 'wb') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)

        # return transcribe_audio(filename)  # Call your transcription function here (index function)
        return filename
        
    return HttpResponse("No file uploaded", status=400)
import subprocess

def preprocess_audio(input_file, output_file):
    command = [
        "ffmpeg",
        "-i", input_file,
        "-af", "highpass=f=200,lowpass=f=3000,loudnorm,acompressor",
        "-ar", "16000",
        "-ac", "1",
        output_file
    ]
    try:
        subprocess.run(command, check=True)
        print(f"Preprocessed audio saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during audio preprocessing: {e}")

# # Example usage:
# input_audio = "input_audio.mp3"
# output_audio = "preprocessed_audio.wav"
# preprocess_audio(input_audio, output_audio)
