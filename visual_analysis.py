import cv2
import os
import statistics
from keras.models import model_from_json

class EmotionPredictor:
    def __init__(self, model_json_path, model_weights_path, labels):
        self.model = self.load_model(model_json_path, model_weights_path)
        self.labels = labels

    def load_model(self, model_json_path, model_weights_path):
        json_file = open(model_json_path, "r")
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights(model_weights_path)
        return model

    def extract_features(self, image):
        """Preprocesses the image for emotion detection.

        input:
            image: A NumPy array representing the image.

        Returns:
            A NumPy array representing the preprocessed image.
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Resize the image to the model's expected input size
        target_size = (48, 48)
        image = cv2.resize(image, target_size)

        # Normalize pixel values between 0 and 1
        image = image.astype('float32') / 255.0

        # Reshape to add a batch dimension for model prediction
        image = image.reshape((1, target_size[0], target_size[1], 1))

        return image

    def predict_emotion(self, image_path):
        """Predicts the emotion in an image using the loaded model.

        Input:
            image_path: The path to the image file.

        Returns:
            A tuple containing the predictions, the index of the max probability, and the predicted emotion label.
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None, None, None

        # Preprocess the image
        preprocessed_image = self.extract_features(image)

        # Make predictions
        predictions = self.model.predict(preprocessed_image)
        max_prob_index = predictions.argmax()
        sentiment_score = self.convert_label_to_score(self.labels[max_prob_index])

        return predictions, max_prob_index, sentiment_score

    def convert_label_to_score(self, label):
        # Define a mapping from label to sentiment score
        label_to_score = {
            'angry': -1,
            'disgust': -1,
            'fear': -1,
            'happy': 1,
            'neutral': 0,
            'sad': -1,
            'surprise': 1
        }
        return label_to_score.get(label, 0)  # Default to 0 if label not found

    def process_images(self, image_folder):
        prob_images = []

        for filename in os.listdir(image_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                image_path = os.path.join(image_folder, filename)
                predictions, max_prob, sentiment_score = self.predict_emotion(image_path)

                if sentiment_score:
                    print(f"Image: {filename} - Predicted probabilities: {predictions} Max Prob: {max_prob} Sentiment Score: {sentiment_score}")
                    prob_images.append(sentiment_score)

        if prob_images:
            prob_mode = int(statistics.mode(prob_images))
            prob_mean = int(round(statistics.mean(prob_images)))

            overall_result_mode = self.convert_score_to_label(prob_mode)
            overall_result_mean = self.convert_score_to_label(prob_mean)

            print("Mode :", prob_mode, " Mean : ", prob_mean)
            print("Overall result mode is : ", overall_result_mode)
            print("Overall result mean is : ", overall_result_mean)
        else:
            print("No valid predictions found.")

        return prob_mode, prob_mean, overall_result_mode, sentiment_score

    def convert_score_to_label(self, score):
        # Define a mapping from sentiment score to label
        score_to_label = {
            -1: 'Negative',
            0: 'Neutral',
            1: 'Positive'
        }
        return score_to_label.get(score, 'Neutral')  # Default to 'Neutral' if score not found
