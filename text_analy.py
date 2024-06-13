
from transformers import BertTokenizer, BertForSequenceClassification
from torch.nn.functional import softmax

class TextSentimentAnalyzer:
    def __init__(self, model_name="nlptown/bert-base-multilingual-uncased-sentiment"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)
        self.sentiment_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]

    def analyze_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        logits = outputs.logits
        probabilities = softmax(logits, dim=1)
        probabilities_array = probabilities.detach().numpy()[0]  # Access the first element and flatten
        probabilities_list = probabilities_array.tolist()
        positive_score = probabilities[:, 3].item() + probabilities[:, 4].item()
        negative_score = probabilities[:, 0].item() + probabilities[:, 1].item()
        sentiment_score = (positive_score - negative_score) / (positive_score + negative_score)
        return sentiment_score, probabilities_list

    def read_text_from_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        return text

    def sentiment_analysis_on_file(self, file_path):
        text = self.read_text_from_file(file_path)
        sentiment_score, probabilities_list = self.analyze_sentiment(text)
        if sentiment_score == 0:
            sentiment_type = "Neutral"
        elif sentiment_score > 0:
            sentiment_type = "Positive"
        elif sentiment_score < 0:
            sentiment_type = "Negative"
        return sentiment_score, sentiment_type, probabilities_list