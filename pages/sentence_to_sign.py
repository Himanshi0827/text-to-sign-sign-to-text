import cv2
import os
import numpy as np
import mediapipe as mp
from keras.models import load_model
from googletrans import Translator
from gtts import gTTS
import streamlit as st
import time
from moviepy.editor import VideoFileClip, concatenate_videoclips
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play



def concatenate_and_play_videos(video_paths):
    clips = []
    for video_path in video_paths:
        clip = VideoFileClip(video_path)
        clip = clip.subclip(0, clip.duration)  # Ensure the clip is using the full duration
        clips.append(clip)

    # Ensure all clips have the same fps
    fps = clips[0].fps
    for clip in clips:
        if clip.fps != fps:
            clip = clip.set_fps(fps)
    
    final_clip = concatenate_videoclips(clips, method="compose")
    
    final_clip_path = "output.mp4"
    final_clip.write_videofile(final_clip_path, codec="libx264", audio_codec="aac")

    # Display the final concatenated video
    with open(final_clip_path, 'rb') as video_file:
        video_bytes = video_file.read()
        st.video(video_bytes)



# Function to check and collect video paths based on words
def check_and_collect_videos(sentence):
    words = sentence.split()  # Split the sentence into words
    video_paths = []

    for word in words:
        video_path = f"Video/{word}.mp4"  # Assuming videos are named after the words with .mp4 extension
        if os.path.exists(video_path):
            video_paths.append(video_path)
        else:
            st.write(f"Video not found for: {word}")

    return video_paths



def sentence_to_sign():
    sentence = st.text_input("Enter a sentence:")
    if sentence:
            video_paths = check_and_collect_videos(sentence)
            if video_paths:
                concatenate_and_play_videos(video_paths)
            else:
                st.write("No videos found to play.")

sentence_to_sign()