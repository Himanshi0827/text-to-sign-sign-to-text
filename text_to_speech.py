import streamlit as st
import os
import time

# Function to play video
def play_video(video_path):
    video_file = open(video_path, 'rb')
    video_bytes = video_file.read()
    st.video(video_bytes)
    video_file.close()

# Function to check and play videos based on words
def check_and_play_videos(sentence):
    words = sentence.split()  # Split the sentence into words

    for word in words:
        video_path = f"{word}.mp4"  # Assuming videos are named after the words with .mp4 extension
        if os.path.exists(video_path):
            st.write(f"Playing video for: {word}")
            play_video(video_path)
            time.sleep(2)  # Wait 2 seconds before playing the next video
        else:
            st.write(f"Video not found for: {word}")

# Streamlit app
st.title("Sentence to Video Player")

sentence = st.text_input("Enter a sentence:")

if sentence:
    check_and_play_videos(sentence)
