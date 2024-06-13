import speech_recognition as sr
from textblob import TextBlob

class AudioSentimentAnalyzer:
    def __init__(self):
        pass

    def transcribe_audio_to_text(self, audio_path):
        recognizer = sr.Recognizer()

        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)

        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio.")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

    def analyze_sentiment(self, text):
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        if sentiment_score < 0:
            sentiment_type = "Negative"
        elif sentiment_score > 0:
            sentiment_type = "Positive"
        else:
            sentiment_type = "Neutral"
        return sentiment_score, sentiment_type

    def analyze_audio_sentiment(self, audio_path):
        transcribed_text = self.transcribe_audio_to_text(audio_path)

        if transcribed_text is not None:
            sentiment_score, sentiment_type = self.analyze_sentiment(transcribed_text)
            print(f"The sentiment analysis for the audio file is {sentiment_type}")
            return sentiment_score, sentiment_type
        else:
            return None