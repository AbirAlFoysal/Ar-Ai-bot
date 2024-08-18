from django.http import JsonResponse, FileResponse, HttpResponse, StreamingHttpResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
import os, sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import speech_recognition as sr
import pyttsx3
from moviepy.editor import VideoFileClip, AudioFileClip, concatenate_videoclips
from skimage.transform import resize
from skimage import img_as_ubyte
import imageio

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..') 
from avatar import *
from avatar import main as create_avatar 

def stream_video(request):
    video_path = os.path.join('output_video.mp4')
    
    if os.path.exists(video_path):
        return FileResponse(open(video_path, 'rb'), content_type='video/mp4')
    else:
        raise Http404("Video not found")









def get_video(request):
    try:
        file_path = 'generated.mp4'  # Path to your video file
        if not os.path.exists(file_path):
            return HttpResponse(status=404, content='Video file not found.')
        
        response = FileResponse(open(file_path, 'rb'), content_type='video/mp4')
        response['Content-Disposition'] = 'inline; filename="generated.mp4"'
        return response
    except Exception as e:
        print(f"Error in get_video view: {e}")
        return HttpResponse(status=500, content=f"Internal Server Error: {e}")






def merge_audio_video(audio_path, video_path, output_path):
    # Check if the output file already exists
    if os.path.exists(output_path):
        os.remove(output_path)  # Remove the existing file

    # Load the video and audio files
    try:
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Get the duration of the audio and video
    audio_duration = audio.duration
    video_duration = video.duration

    print(f"Audio duration: {audio_duration} seconds")
    print(f"Video duration: {video_duration} seconds")

    if audio_duration < video_duration:
        # Trim the video to match the audio duration
        video = video.subclip(0, audio_duration)
    elif audio_duration > video_duration:
        # Repeat the video to match the audio duration
        clips = []
        while sum([clip.duration for clip in clips]) < audio_duration:
            clips.append(video)
        video = concatenate_videoclips(clips).subclip(0, audio_duration)

    # Set the audio to the video
    video = video.set_audio(audio)

    # Write the final video file
    try:
        video.write_videofile(output_path, codec="libx264", audio_codec="aac", temp_audiofile='temp-audio.m4a', remove_temp=True)
    except Exception as e:
        print(f"Error writing video file: {e}")














import pyttsx3
import os

def text_to_speech(text):
    filename="speech"
    file_format='mp3'
    # Add the file extension to the filename
    filename_with_extension = f"{filename}.{file_format}"
    
    # Check if the file already exists
    if os.path.exists(filename_with_extension):
        os.remove(filename_with_extension)  # Remove the existing file

    # Initialize the pyttsx3 engine
    engine = pyttsx3.init()

    # Set properties before adding anything to speak
    engine.setProperty('rate', 150)    # Speed of speech
    engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)

    # Save the speech to an audio file
    engine.save_to_file(text, filename_with_extension)

    # Run the engine 
    engine.runAndWait()



def get_video(request):
    try:
        file_path = 'generated.mp4'  # Path to your video file
        if not os.path.exists(file_path):
            return HttpResponse(status=404, content='Video file not found.')
        
        response = FileResponse(open(file_path, 'rb'), content_type='video/mp4')
        response['Content-Disposition'] = 'inline; filename="generated.mp4"'
        return response
    except Exception as e:
        print(f"Error in get_video view: {e}")
        return HttpResponse(status=500, content=f"Internal Server Error: {e}")


@csrf_exempt
def submit_voice(request):
    if request.method == 'POST':
        audio_file = request.FILES.get('audio')
        if (audio_file):
            try:
                # Save the audio file temporarily
                file_path = default_storage.save('temp_audio.wav', ContentFile(audio_file.read()))
                recognizer = sr.Recognizer()

                with sr.AudioFile(file_path) as source:
                    audio_data = recognizer.record(source)
                    text = recognizer.recognize_google(audio_data)  # Convert audio to text
                    print(f"Converted text: {text}")

                    # Generate response using the chatbot
                    response_text = generate_response(text)
                    text_to_speech(response_text)

                    # Handle video merging and playing it on the webpage
                    merge_audio_video("speech.mp3", "generated.mp4", "output_video.mp4")
                    print(f"Generated response: {response_text}")

                os.remove(file_path)  # Clean up the saved file

                # Return a success response and trigger video playback
                return JsonResponse({'status': 'success', 'text': text, 'response': response_text})
            except sr.UnknownValueError:
                print("Could not understand audio")
                return JsonResponse({'status': 'error', 'message': 'Could not understand audio'})
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                return JsonResponse({'status': 'error', 'message': f'Could not request results; {e}'})
            except Exception as e:
                print(f"Unexpected error: {e}")
                return JsonResponse({'status': 'error', 'message': f'Unexpected error: {e}'})
        else:
            print("No audio data received")
            return JsonResponse({'status': 'error', 'message': 'No audio data'})
    else:
        print("Invalid request method")
        return JsonResponse({'status': 'error', 'message': 'Invalid request method'})





# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import base64
from PIL import Image
from io import BytesIO

@csrf_exempt
def submit_picture(request):
    if request.method == 'POST':
        image_data = request.POST.get('image')
        if image_data:
            format, imgstr = image_data.split(';base64,') 
            image = base64.b64decode(imgstr)
            image = Image.open(BytesIO(image))

            # Crop the image to a square from the center
            width, height = image.size
            new_dim = min(width, height)
            left = (width - new_dim) / 2
            top = (height - new_dim) / 2
            right = (width + new_dim) / 2
            bottom = (height + new_dim) / 2

            image = image.crop((left, top, right, bottom))

            # Resize to 680x676
            image = image.resize((680, 676), Image.Resampling.LANCZOS)

            # Save the image
            image.save('submitted_image.png')

            # Process the image further if needed
            create_avatar('submitted_image.png')

            return JsonResponse({'status': 'success', 'message': 'Image received'})
        return JsonResponse({'status': 'error', 'message': 'No image data'})
    return JsonResponse({'status': 'error', 'message': 'Invalid request method'})

# @csrf_exempt
# def submit_picture(request):
#     if request.method == 'POST':
#         image_data = request.POST.get('image')
#         if image_data:

#             format, imgstr = image_data.split(';base64,') 
#             image = base64.b64decode(imgstr)
#             image = Image.open(BytesIO(image))
            
#             image.save('submitted_image.png')

#             create_avatar('submitted_image.png')


#             return JsonResponse({'status': 'success', 'message': 'Image received'})
#         return JsonResponse({'status': 'error', 'message': 'No image data'})
#     return JsonResponse({'status': 'error', 'message': 'Invalid request method'})


model_name = "EleutherAI/pythia-410m-deduped"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)




@csrf_exempt
def chat(request):
    if request.method == 'POST':
        input_text = str(request.POST.get('message'))
        print(f"Received message: {input_text}")  # Logging the received message
        if input_text:
            response_text = generate_response(input_text)
            text_to_speech(response_text)

            # Handling video merging and playing it on the webpage
            merge_audio_video("speech.mp3", "generated.mp4", "output_video.mp4")

            print(f"Generated response: {response_text}")  # Logging the response

            return JsonResponse({'response': response_text})
        return JsonResponse({'response': 'No input text provided.'})
    return render(request, 'home.html')



def generate_response(input_text):
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(
        inputs, 
        max_length=50, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.6, 
        top_p=0.8, 
        repetition_penalty=1.2
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated response: {response}")  
    clean_text =  str(response.strip()).replace(input_text,"")
    return clean_text

 
# def second_page(request):
#     return render(request, 'second.html')







import cv2
from fer import FER
from django.http import StreamingHttpResponse
from django.shortcuts import render
import threading

def emotion_detection_stream():
    cap = cv2.VideoCapture(0)
    detector = FER(mtcnn=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            result = detector.detect_emotions(frame)
            if result:
                emotion, score = detector.top_emotion(frame)
                if emotion:
                    emotion_text = f"{emotion} ({score*100}%)"
                else:
                    emotion_text = "Unknown"
            else:
                emotion_text = "Unknown"
        except Exception as e:
            emotion_text = "Error"
            print(f"Error: {e}")

        cv2.putText(frame, f'Emotion: {emotion_text}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        _, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def video_feed(request):
    return StreamingHttpResponse(emotion_detection_stream(),
                                 content_type='multipart/x-mixed-replace; boundary=frame')
