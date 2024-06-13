from diffusion import DiffusionStage
from text_analy import TextSentimentAnalyzer
from audio_analy import AudioSentimentAnalyzer
from visual_analysis import EmotionPredictor

def combine_sentiments(text_score, audio_score, visual_score):
    # Convert scores to float if they are strings
    try:
        text_score = float(text_score)
    except ValueError:
        text_score = 0.0
    try:
        audio_score = float(audio_score)
    except ValueError:
        audio_score = 0.0
    try:
        visual_score = float(visual_score)
    except ValueError:
        visual_score = 0.0

    # Combine the sentiment scores from all modalities (assuming equal weights for now)
    overall_score = (text_score + audio_score + visual_score) / 3

    # Determine the overall sentiment type
    if overall_score < 0:
        overall_sentiment = "Negative"
    elif overall_score > 0:
        overall_sentiment = "Positive"
    else:
        overall_sentiment = "Neutral"

    return overall_score, overall_sentiment

def main(video):
    # Stage 1: DIFFUSION
    video_path = video
    converter = DiffusionStage(video_path)
    result_text = converter.diffusion()

    print("Transcription from Video:")
    print(result_text)
    print(f"Transcription saved to: {converter.output_file}")

    # Stage 2: SENTIMENT ANALYSIS OF INDIVIDUAL MODALITIES
    # Part 1: Text modality sentiment analysis
    file_path = 'diff_text_output.txt'
    text_analyzer = TextSentimentAnalyzer()
    text_score, _,  _ = text_analyzer.sentiment_analysis_on_file(file_path)
    print(f"the final sentiment score of text after analysis is {text_score}")

    # Part 2: Audio modality sentiment analysis
    audio_path = 'diff_audio_output.wav'
    audio_analyzer = AudioSentimentAnalyzer()
    audio_score, _ = audio_analyzer.analyze_audio_sentiment(audio_path)
    print(f"the final sentiment score of audio after analysis is {audio_score}")

    # Part 3: Visual modality sentiment analysis
    model_json_path = "emotiondetector.json"
    model_weights_path = "emotiondetector.h5"
    labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

    visual_analyzer = EmotionPredictor(model_json_path, model_weights_path, labels)
    _, _, _, visual_score = visual_analyzer.process_images("diff_image_output")
    print(f"the final sentiment score of visual after analysis is {visual_score}")

    # Combine the sentiment scores from all modalities
    overall_score, overall_sentiment = combine_sentiments(text_score, audio_score, visual_score)
    print(f"the final sentiment score after combining result is {overall_sentiment}, the type of sentiment is{overall_sentiment}")
    return text_score, audio_score, visual_score, overall_score, overall_sentiment
