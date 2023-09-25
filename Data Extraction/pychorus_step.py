import os
import pychorus as pc
import pandas as pd
import soundfile as sf

def extract_chorus(row):
    # Read the audio file
    audio_file = str(row['file_path']) + ".mp3"
    try:
        data, samplerate = sf.read(audio_file)
    except Exception as e:
        print(f"Error reading audio file: {audio_file}")
        print(f"Error message: {str(e)}")
        return pd.Series({'chorus_path': float('nan')})

    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        return pd.Series({'chorus_path': float('nan')})
    
    # Get the chromagram
    try:
        chroma, _, _, song_length_sec = pc.create_chroma(audio_file)
    except Exception as e:
        print(f"Error creating chromagram for file: {audio_file}")
        print(f"Error message: {str(e)}")
        return pd.Series({'chorus_path': float('nan')})

    # Set the clip length (in seconds)
    clip_length = 15

    # Find the chorus section
    chorus_start = pc.find_chorus(chroma, samplerate, song_length_sec, clip_length)

    if chorus_start is None:
        print("No chorus detected for song:", row['title'])
        return pd.Series({'chorus_path': float('nan')})

    # Set the duration of the chorus (in seconds)
    chorus_duration = 15

    # Calculate start and end time of chorus
    start_time = int(chorus_start * samplerate)
    end_time = start_time + int(chorus_duration * samplerate)

    # Extract the chorus section
    chorus = data[start_time:end_time]

    # Check if chorus file already exists
    chorus_file = f"{row['title']}_chorus.wav"
    chorus_path = os.path.join('choruss', chorus_file)
    if os.path.exists(chorus_path):
        print(f"Chorus file already exists at {chorus_path}.")
    else:
        # Write chorus to WAV file
        try:
            sf.write(chorus_path, chorus, samplerate)
        except Exception as e:
            print(f"Error writing chorus to file: {chorus_path}")
            print(f"Error message: {str(e)}")
            return pd.Series({'chorus_path': float('nan')})
        print(f"Chorus extracted and saved as {chorus_path}.")

    # Add extracted chorus path to the dataframe
    return pd.Series({'chorus_path': chorus_path})

# Load the dataframe with song information
df = pd.read_csv('songs_with_file_paths.csv')

# Extract choruses for each song and add chorus file path to the dataframe
df = df.apply(extract_chorus, axis=1).join(df)

# Save the updated dataframe to a CSV file
df.to_csv('songs_with_chorus_paths.csv', index=False)