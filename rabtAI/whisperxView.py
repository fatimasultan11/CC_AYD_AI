import os
import whisperx
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import spacy
import torch

# Initialize models
whisper_model = whisperx.load_model("base",device='cpu',compute_type='int8')
sentiment_analyzer = pipeline("sentiment-analysis")
nlp = spacy.load("en_core_web_trf")
# "en_core_web_sm"

def transcribe_audio_with_speakers(audio_file):
    """Transcribes audio and identifies speaker labels using WhisperX."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisperx.load_model("medium", device,compute_type='int8')
    
    # Step 1: Transcription
    result = model.transcribe(audio_file)
    
    # Step 2: Align transcription with speaker diarization
    diarize_model = whisperx.DiarizationPipeline(device=device)
    diarized_result = diarize_model(audio_file, result["segments"])
    segments = diarized_result["segments"]
    
    # Combine transcriptions with speaker info
    sentences_with_speakers = []
    for segment in segments:
        speaker = segment["speaker"]
        text = segment["text"]
        sentences_with_speakers.append(f"{speaker}: {text}")
    
    return " ".join(sentences_with_speakers)

def segment_text(text):
    """Segments text into sentences."""
    return sent_tokenize(text)

def analyze_sentiment(sentences):
    """Performs sentiment analysis on a list of sentences."""
    return [{"sentence": sent, "sentiment": sentiment_analyzer(sent)[0]} for sent in sentences]

def extract_keywords(text):
    """Extracts keywords using SpaCy."""
    doc = nlp(text)
    keywords = [chunk.text for chunk in doc.noun_chunks]
    return list(set(keywords))  # Remove duplicates

def process_audio(audio_file):
    """Processes an audio file to transcribe, segment, and analyze."""
    # Transcribe audio with speaker information
    transcription_with_speakers = transcribe_audio_with_speakers(audio_file)
    print("Transcription with Speakers Complete.")
    
    # Segment transcription into sentences
    sentences = segment_text(transcription_with_speakers)
    print(f"Segmented into {len(sentences)} sentences.")
    
    # Perform sentiment analysis
    sentiments = analyze_sentiment(sentences)
    print("Sentiment Analysis Complete.")
    
    # Perform keyword extraction
    keywords = extract_keywords(transcription_with_speakers)
    print("Keyword Extraction Complete.")
    
    return {
        "transcription_with_speakers": transcription_with_speakers,
        "sentiments": sentiments,
        "keywords": keywords
    }

# Example usage
if __name__ == "__main__":
    audio_file = "data/in-4238020121-03122532200-20240706-111915-1720246755.17908.wav"
    if not os.path.exists(audio_file):
        print("File not found")
        exit()
    results = process_audio(audio_file)
    
    print("\nTranscription with Speakers:\n", results["transcription_with_speakers"])
    print("\nSentiment Analysis:\n", results["sentiments"])
    print("\nExtracted Keywords:\n", results["keywords"])
