import os
import cv2
from keras.models import model_from_json
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, jsonify, redirect, session, url_for, Response

# Define the upload folder and allowed video file extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv', 'flv', 'wmv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'  # Needed for session management


# Create the upload folder if it doesn't exist
def create_upload_folder():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


create_upload_folder()

json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load the face cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def generate_frames():
    webcam = cv2.VideoCapture(0)
    while True:
        success, frame = webcam.read()
        if not success:
            break
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (p, q, r, s) in faces:
                image = gray[q:q+s, p:p+r]
                cv2.rectangle(frame, (p, q), (p+r, q+s), (255, 0, 0), 2)
                image = cv2.resize(image, (48, 48))
                img = extract_features(image)
                pred = model.predict(img)
                prediction_label = labels[pred.argmax()]
                cv2.putText(frame, '% s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('baseapp.html')  # Replace with your actual template name


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request!'}), 400

    uploaded_file = request.files['file']

    if uploaded_file.filename == '':
        return jsonify({'message': 'No file selected!'}), 400

    if allowed_file(uploaded_file.filename):
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            uploaded_file.save(filepath)
            # Call sentiment analysis program (replace with the actual path and function)
            from main import main
            res_text, res_audio, res_visual, sentiment_result, sentiment_type= main(filepath)

            # Store the result in the session
            session['results'] = {
                'res_text': res_text,
                'res_audio': res_audio,
                'res_visual': res_visual,
                'sentiment_result': sentiment_result,
                'sentiment_type': sentiment_type
            }

            return redirect(url_for('show_results'))

        except Exception as e:
            return jsonify({'message': f"File upload or processing failed: {str(e)}"}), 500
    else:
        return jsonify({'message': 'File type not allowed!'}), 400


@app.route('/results')
def show_results():
    results = session.get('results', None)
    if results:
        return render_template('results.html', results=results)
    else:
        return redirect(url_for('index'))


@app.route('/realtime')
def realtime():
    return render_template('realtimeanalysis.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)