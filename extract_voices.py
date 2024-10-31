"""
    This script uses the `pyannote.audio` library to perform speaker diarization on audio files,
    separating different voices (speakers) in an audio file. Each voice segment is saved as a separate audio file 
    in a specified output folder. This process includes Voice Activity Detection (VAD).
    
    Overview of the Script:

    1. Diarization Pipeline Loading: 
       The script loads a pre-trained speaker diarization model (`pyannote/speaker-diarization-3.1`) 
       from Hugging Face (https://huggingface.co) using an authentication token.
       !Important! A token is required to use the model. Generate one from Hugging Face; 
       otherwise, the script won’t work.

    2. File Sorting: 
       The audio files in the input folder are sorted in ascending order by file size to streamline processing.

    3. Diarization and Voice Segment Extraction: 
       Each audio file is processed to identify voice segments spoken by different speakers.
       These segments are saved as individual audio files, named according to the speaker and segment order.

    4. Processing Progress: 
       The `tqdm` library displays the progress of file processing in real time.

    5. Output Folder Creation: 
       If the specified output folder does not exist, the script automatically creates it.

    Function Parameters for `extract_voices`:
    - `input_folder_path`: Path to the folder containing the original audio files in WAV format.
    - `output_folder_path`: Path to the folder where separated voice segments will be saved.

    *** Instructions for Using this Script ***
    1. Generate a token to use the model on Hugging Face (model link: https://huggingface.co/pyannote/speaker-diarization-3.1).
    2. Enter the token in the `pipeline` ("enter-your-token-here").
    3. Set the path to the folder with original audio files in "input_folder_path".
    4. Set the path to the folder where the identified speaker-specific audio files will be saved.

    !!! NOTE !!!
        This program uses an automatic Voice Activity Detector. If you have three speakers (1,2,3) speaking in this order:
        1-3-2-1-2-3-1, seven different audio files will be generated.

    !!! Warning !!!
        The Hugging Face model was trained on mono audio files with a 16kHz sample rate.
        During diarization, the model automatically converts files to mono and resamples them to 16kHz.

    !!! Updates and Modifications !!!
        The model used can be replaced while keeping the rest of the code unchanged.

    Python Version: 3.9.6

    Library Versions:
        pyannote.audio - Version: 3.3.1
        tqdm - Version: 4.66.4
        pydub - Version: 0.25.1

    !!! Make sure all mentioned libraries are installed to run the script successfully !!!
"""

import os
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from tqdm import tqdm
from pydub import AudioSegment

# Load pre-trained diarization pipeline from Hugging Face using authentication token
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token="enter-your-token-here"
)

# Define input and output folder paths
input_folder_path = "path-to-original-audio-folder"  
output_folder_path = "path-to-output-folder-for-separated-voices" 

def extract_voices(input_folder_path, output_folder_path):
    """
    Extracts voice segments from audio files, saves them in a specified output folder.

    Args:
        input_folder_path (str): Path to the folder containing the original audio files in WAV format.
        output_folder_path (str): Path to the folder where separated voice segments will be saved.
    """
    # Create output directory if it does not exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # Sort audio files in input folder by file size (ascending)
    audio_files = sorted([f for f in os.listdir(input_folder_path) if f.endswith('.wav')],
                         key=lambda f: os.path.getsize(os.path.join(input_folder_path, f)))

    # Process each audio file with a progress bar
    with tqdm(total=len(audio_files), desc="Processing files") as pbar:
        for audio_file in audio_files:
            audio_path = os.path.join(input_folder_path, audio_file)
            
            # Run diarization pipeline with progress hook
            with ProgressHook() as hook:
                diarization = pipeline(audio_path)

            # Load the audio file using pydub
            audio = AudioSegment.from_wav(audio_path)

            # Extract each segment, name it based on speaker and segment order, and save it
            base_filename = os.path.splitext(audio_file)[0]
            segment_count = 0
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # Extract segment based on start and end times
                segment = audio[turn.start * 1000: turn.end * 1000]  # Convert to milliseconds
                speaker_file = os.path.join(output_folder_path, f"{base_filename}_speaker_{speaker}_segment_{segment_count}.wav")
                segment.export(speaker_file, format="wav")
                segment_count += 1

            # Update progress bar
            pbar.update(1)

    print("Voice separation from audio files completed successfully.")
