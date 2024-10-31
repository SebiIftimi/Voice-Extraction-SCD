"""
    This program is an enhanced version of "extract_voices.py". Unlike the original, where each speaker segment creates a new file,
    this version combines all segments from the same speaker into a single file. 
    For example, if speakers appear in the order: 1 2 1 3 1 2 2, the program will identify all unique speakers
    and create only 3 files instead of 7. The files will contain combined segments as follows:
        - File 1: segments for speaker 1 (e.g., 111)
        - File 2: segments for speaker 2 (e.g., 222)
        - File 3: segments for speaker 3 (e.g., 3)
"""

import os
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from tqdm import tqdm
from pydub import AudioSegment

# Load pre-trained diarization pipeline with authentication token from Hugging Face
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="your-huggingface-token"
)

# Define input and output folder paths
input_folder_path = "path-to-original-audio-folder"  
output_folder_path = "path-to-output-folder-for-diarized-voices"  

# Create output directory if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# List all WAV files in the input folder
audio_files = [f for f in os.listdir(input_folder_path) if f.endswith('.wav')]

# Process each audio file with a progress bar
with tqdm(total=len(audio_files), desc="Processing files") as pbar:
    for audio_file in audio_files:
        audio_path = os.path.join(input_folder_path, audio_file)
        
        # Run diarization pipeline with progress hook
        with ProgressHook() as hook:
            diarization = pipeline(audio_path)

        # Load audio file with pydub
        audio = AudioSegment.from_wav(audio_path)
        speaker_segments = {}

        # Iterate through each diarized segment and group by speaker
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Initialize audio segment for new speakers
            if speaker not in speaker_segments:
                speaker_segments[speaker] = AudioSegment.empty()
            
            # Extract segment and add it to the corresponding speaker's file
            segment = audio[turn.start * 1000: turn.end * 1000]  # Convert to milliseconds
            speaker_segments[speaker] += segment

        # Save combined segments for each speaker
        base_filename = os.path.splitext(audio_file)[0]
        for speaker, audio_segment in speaker_segments.items():
            speaker_file = os.path.join(output_folder_path, f"{base_filename}_speaker_{speaker}.wav")
            
            # Check if a file for this speaker already exists, and if so, append to it
            if not os.path.exists(speaker_file):
                audio_segment.export(speaker_file, format="wav")
            else:
                # Append new segment to existing file
                existing_audio = AudioSegment.from_wav(speaker_file)
                combined_audio = existing_audio + audio_segment
                combined_audio.export(speaker_file, format="wav")

        # Update progress bar after processing each file
        pbar.update(1)

print("Processing complete.")
