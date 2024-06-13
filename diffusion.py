import os
import moviepy.editor as mp
import speech_recognition as sr


class DiffusionStage:
    def __init__(self, video_path):
        self.video_path = video_path
        self.output_file = 'diff_text_output.txt'
        self.audio_path = 'diff_audio_output.wav'

    def extract_frames(self, imgdir):
        if not os.path.exists(imgdir):
            os.makedirs(imgdir)
        clip = mp.VideoFileClip(self.video_path)
        duration = int(clip.duration)
        times = [t for t in range(0, duration)]
        for t in times:
            imgpath = os.path.join(imgdir, '{}.png'.format(int(t)))
            clip.save_frame(imgpath, t)

    def extract_audio(self):
        video_clip = mp.VideoFileClip(self.video_path)
        audio_clip = video_clip.audio
        audio_clip.write_audiofile(self.audio_path, codec='pcm_s16le')

    def transcribe_audio_to_text(self):
        recognizer = sr.Recognizer()
        with sr.AudioFile(self.audio_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
        return text

    def save_text_to_file(self, text):
        with open(self.output_file, 'w') as file:
            file.write(text)

    def diffusion(self):
        self.extract_frames("diff_image_output")
        self.extract_audio()
        text = self.transcribe_audio_to_text()
        self.save_text_to_file(text)
        return text

