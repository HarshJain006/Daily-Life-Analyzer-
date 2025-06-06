import streamlit as st
import numpy as np
import librosa
import soundfile as sf
import tempfile
import uuid
import os
import logging
import time
import speech_recognition as sr
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Set page config as the first Streamlit command
st.set_page_config(page_title="LifeVibe Insights", layout="centered")

# Custom CSS for enhanced UI with improved color scheme
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        body {
            background: linear-gradient(135deg, #d1e8ff 0%, #c3f8e6 100%);
            font-family: 'Poppins', sans-serif;
        }
        .main {
            max-width: 800px;
            margin: 0 auto;
            padding: 2rem;
            background: #ffffff;
            border-radius: 1rem;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            transition: transform 0.3s ease;
        }
        .main:hover {
            transform: translateY(-5px);
        }
        .stRadio > label {
            font-size: 1.1rem;
            color: #1e293b;
            margin-right: 1rem;
        }
        .stRadio > div {
            display: flex;
            justify-content: center;
            gap: 1.5rem;
        }
        .stButton > button {
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 0.5rem;
            border: none;
            font-size: 1.1rem;
            transition: transform 0.2s ease, background 0.3s ease;
            width: 100%;
        }
        .stButton > button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #8b5cf6, #3b82f6);
        }
        .stAudio, .stFileUploader {
            border-radius: 0.5rem;
            border: 1px solid #d4d4d8;
        }
        .stMarkdown h1 {
            color: #1e40af;
            font-weight: 700;
            text-align: center;
            margin-bottom: 1rem;
            background: #eff6ff;
            padding: 1rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        .stMarkdown h2 {
            color: #1e293b;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .stMarkdown p {
            color: #475569;
            text-align: center;
            margin-bottom: 2rem;
        }
        .summary {
            background: #e0f2fe;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid #3b82f6;
        }
        .feedback {
            background: #ecfdf5;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1.5rem;
            border-left: 4px solid #10b981;
        }
        .history {
            background: #f5f3ff;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #8b5cf6;
        }
        .history ul {
            list-style-type: disc;
            padding-left: 1.5rem;
            margin: 0;
        }
        .history li {
            color: #475569;
            margin-bottom: 0.5rem;
        }
        .footer {
            text-align: center;
            color: #6b7280;
            margin-top: 2rem;
            font-size: 0.9rem;
        }
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(10px); }
            100% { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize speech recognizer
recognizer = sr.Recognizer()
recognizer.energy_threshold = 300
recognizer.dynamic_energy_threshold = True


# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"),) # Replace with your actual Groq API key


# Initialize session state
if not hasattr(st.session_state, 'initialized'):
    st.session_state.initialized = True
    st.session_state.recorded_audio = None
    st.session_state.recorded_audio_path = None
    st.session_state.transcription_history = []

# Audio processing function
def process_audio(audio, sample_rate=16000):
    if audio is None:
        return None, "0.0 seconds", "No audio detected"
    try:
        if isinstance(audio, bytes):  # From st.audio_input
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmpfile.write(audio)
                tmpfile_path = tmpfile.name
            try:
                y, input_sr = sf.read(tmpfile_path)
                if len(y) == 0:
                    return None, "0.0 seconds", "Empty audio data from microphone"
                y = y.astype(np.float32)
                if len(y.shape) == 2:
                    y = np.mean(y, axis=1)
                if input_sr != sample_rate:
                    y = librosa.resample(y, orig_sr=input_sr, target_sr=sample_rate)
            finally:
                if os.path.exists(tmpfile_path):
                    os.remove(tmpfile_path)
        else:  # From file upload
            if hasattr(audio, 'name'):
                ext = os.path.splitext(audio.name)[1].lower()
                if ext not in ['.wav', '.mp3']:
                    return None, "0.0 seconds", f"Unsupported file format: {ext}. Please upload WAV or MP3."
            temp_file = f"temp_audio_{uuid.uuid4()}{ext if hasattr(audio, 'name') else '.wav'}"
            with open(temp_file, "wb") as f:
                f.write(audio.read())
            try:
                y, input_sr = sf.read(temp_file)
                if len(y) == 0:
                    return None, "0.0 seconds", "No audio data found in file"
                y = y.astype(np.float32)
                if len(y.shape) == 2:
                    y = np.mean(y, axis=1)
                if input_sr != sample_rate:
                    y = librosa.resample(y, orig_sr=input_sr, target_sr=sample_rate)
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        audio_duration = len(y) / sample_rate
        audio_duration_str = f"{audio_duration:.2f} seconds"
        if np.max(np.abs(y)) > 0:
            y /= np.max(np.abs(y))  # Normalize
        return (sample_rate, y), audio_duration_str, None
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return None, "0.0 seconds", f"Error processing audio: {str(e)}"

# Transcription function
def transcribe_audio(audio_data):
    if audio_data is None:
        st.error("No valid audio data to transcribe.")
        return None
    try:
        sample_rate, y = audio_data
        duration = len(y) / sample_rate
        chunk_duration = 30  # Process in 30-second chunks
        num_chunks = int(np.ceil(duration / chunk_duration))
        transcriptions = []

        for i in range(num_chunks):
            start_sample = int(i * chunk_duration * sample_rate)
            end_sample = int(min((i + 1) * chunk_duration * sample_rate, len(y)))
            chunk = y[start_sample:end_sample]
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                tmpfile_name = tmpfile.name
                sf.write(tmpfile_name, chunk, sample_rate)
                logger.debug(f"Created temp file for chunk {i+1}: {tmpfile_name}")
            
            try:
                with sr.AudioFile(tmpfile_name) as source:
                    audio = recognizer.record(source)
                try:
                    text = recognizer.recognize_google(audio, language="en-IN")
                    if text.strip():
                        transcriptions.append(text)
                    logger.debug(f"Chunk {i+1} transcribed: {text}")
                except sr.UnknownValueError:
                    transcriptions.append("Could not understand audio.")
                    logger.warning(f"Chunk {i+1}: Could not understand audio.")
                except sr.RequestError as e:
                    transcriptions.append(f"Speech service error: {e}")
                    logger.error(f"Chunk {i+1}: Speech service error: {e}")
            finally:
                for _ in range(3):
                    try:
                        if os.path.exists(tmpfile_name):
                            os.remove(tmpfile_name)
                            logger.debug(f"Removed temp file for chunk {i+1}: {tmpfile_name}")
                        break
                    except OSError as e:
                        logger.warning(f"Retry deleting {tmpfile_name}: {str(e)}")
                        time.sleep(0.1)

        if transcriptions:
            combined_text = " ".join([t for t in transcriptions if t and not t.startswith("Speech service error")])
            if combined_text.strip():
                st.session_state.transcription_history.append(combined_text)
                return combined_text
            else:
                st.session_state.transcription_history.append("No understandable audio detected.")
                st.warning("No understandable audio detected.")
                return None
        else:
            st.session_state.transcription_history.append("No audio transcribed.")
            st.warning("No audio transcribed.")
            return None
    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}")
        st.session_state.transcription_history.append(f"Transcription error: {str(e)}")
        st.error(f"Transcription error: {str(e)}")
        return None

# Streamlit UI
st.markdown('<div class="main fade-in">', unsafe_allow_html=True)

# Place headline and description inside the .main div
st.markdown("""
    <h1>🧠 LifeVibe Insights</h1>
    <p>Capture your day with audio and uncover insights with a vibrant analysis!</p>
""", unsafe_allow_html=True)

# Input method selection
st.markdown('<h2>🎙️ Choose Your Input</h2>', unsafe_allow_html=True)
input_method = st.radio(
    "Select input method:",
    ["record", "upload"],
    format_func=lambda x: "Record Audio" if x == "record" else "Upload Audio",
    key="input_method"
)

# Audio input based on selection
if input_method == "record":
    st.markdown('<h2>🎤 Record Your Day</h2>', unsafe_allow_html=True)
    recorded_audio = st.audio_input("Record audio about your day", key="audio_input")
    if recorded_audio:
        st.session_state.recorded_audio = recorded_audio.getvalue()
        st.audio(st.session_state.recorded_audio, format="audio/wav")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmpfile.write(st.session_state.recorded_audio)
            y, input_sr = sf.read(tmpfile.name)
            rms = np.sqrt(np.mean(y**2))
            st.markdown(f'<p class="fade-in">Recorded audio RMS amplitude: {rms:.6f}</p>', unsafe_allow_html=True)
            if rms < 1e-4:
                st.warning("Warning: Recorded audio seems silent!")
            st.session_state.recorded_audio_path = tmpfile.name
        if os.path.exists(st.session_state.recorded_audio_path):
            try:
                os.remove(st.session_state.recorded_audio_path)
                st.session_state.recorded_audio_path = None
            except Exception as e:
                logger.error(f"Error cleaning up temporary file: {str(e)}")
else:
    st.markdown('<h2>📤 Upload Audio</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an audio file (WAV or MP3)", type=["wav", "mp3"], key="upload")
    st.session_state.recorded_audio = None

# Transcribe and analyze button
if st.button("Transcribe & Analyze My Day"):
    audio = st.session_state.recorded_audio if input_method == "record" else uploaded_file
    if audio:
        with st.spinner("Transcribing audio..."):
            processed_audio, duration, error = process_audio(audio)
            if error:
                st.error(error)
            else:
                transcription = transcribe_audio(processed_audio)
                if transcription:
                    with st.spinner("Analyzing your day..."):
                        # First Prompt: Analyze daily story
                        analysis_prompt = f"""
                        You are an LLM model that listens to the user's daily story and gives an analysis. 
                        Based on the following story, summarize:
                        - 🌟 Work done
                        - ✅ Progress made
                        - 🎉 Anything special
                        - 📝 To-do list (suggestions)
                        - 📈 Predicted efficiency of the day (in % and short reason)

                        Here is the user input:
                        '''{transcription}'''
                        """

                        analysis_completion = client.chat.completions.create(
                            messages=[{"role": "user", "content": analysis_prompt}],
                            model="llama3-70b-8192"
                        )

                        analysis = analysis_completion.choices[0].message.content
                        st.markdown('<h2>📋 Your Day Summary</h2>', unsafe_allow_html=True)
                        st.markdown(f'<div class="summary">{analysis}</div>', unsafe_allow_html=True)

                        # Second Prompt: Suggestion and sweet message
                        feedback_prompt = f"""
                        You are a supportive and sweet life coach LLM. Based on the user's day, write a short note (up to 3 lines):
                        - 🧁 A sweet, motivational message
                        - 🌱 One thing they can improve on tomorrow

                        Here is the user's day:
                        '''{transcription}'''
                        """

                        feedback_completion = client.chat.completions.create(
                            messages=[{"role": "user", "content": feedback_prompt}],
                            model="llama3-70b-8192"
                        )

                        feedback = feedback_completion.choices[0].message.content
                        st.markdown('<h2>💌 Feedback for Tomorrow</h2>', unsafe_allow_html=True)
                        st.markdown(f'<div class="feedback">{feedback}</div>', unsafe_allow_html=True)
                if input_method == "record":
                    st.session_state.recorded_audio = None
    else:
        st.error("No audio input provided. Please record or upload an audio file.")

# Display transcription history
if st.session_state.transcription_history:
    st.markdown('<h2>📜 Transcription History</h2>', unsafe_allow_html=True)
    history_html = '<div class="history"><ul>'
    for text in st.session_state.transcription_history:
        history_html += f'<li>{text}</li>'
    history_html += '</ul></div>'
    st.markdown(history_html, unsafe_allow_html=True)

# Footer
st.markdown('<p class="footer">Designed By: Harsh Jain</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)



 
