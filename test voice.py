import speech_recognition as sr
import os
import subprocess

recognizer = sr.Recognizer()

print("\n📂 Audio files in project folder:\n")

audio_files = [
    f for f in os.listdir()
    if f.endswith((".wav", ".mp3", ".ogg"))
]

for i, f in enumerate(audio_files):
    print(f"{i+1}. {f}")

if not audio_files:
    print("❌ No audio files found.")
    exit()

choice = int(input("\nSelect file number to test: ")) - 1
file_path = audio_files[choice]

# ---------------- CONVERT OGG TO WAV ---------------- #
if file_path.endswith(".ogg"):
    print("🔄 Converting OGG → WAV using FFmpeg...")

    wav_file = "converted.wav"

    subprocess.run([
        "ffmpeg",
        "-y",
        "-i", file_path,
        wav_file
    ])

    file_path = wav_file

# ---------------- SPEECH RECOGNITION ---------------- #
print(f"\n🎧 Testing: {file_path}\n")

with sr.AudioFile(file_path) as source:
    audio = recognizer.record(source)

# -------- AUTO LANGUAGE CHECK -------- #
try:
    text = recognizer.recognize_google(audio, language="hi-IN")
    print("🪔 Hindi detected:")
    print(text)

except:
    try:
        text = recognizer.recognize_google(audio, language="mr-IN")
        print("🌿 Marathi detected:")
        print(text)

    except:
        text = recognizer.recognize_google(audio)
        print("🌎 English detected:")
        print(text)
