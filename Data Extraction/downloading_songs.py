import pandas as pd
import os
import yt_dlp

# Dictionary to store file paths for downloaded songs
file_paths = {}

def download_audio(row):
    video_name = row['title']
    save_path = 'audio_files/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Check if the file path for the song already exists in the dictionary
    if video_name in file_paths:
        return file_paths[video_name]

    # Check if the file path for the song already exists in the audio_files directory
    file_name = video_name.replace('/', '_') + '.mp3'
    file_path = os.path.abspath(os.path.join(save_path, file_name))
    if os.path.exists(file_path):
        # Add the file path to the dictionary
        file_paths[video_name] = file_path
        return file_path

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': save_path + file_name,
        'postprocessors': [
            {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192'
            }
        ],
        'ffmpeg_location': 'C:/path_programs_fmpeg/ffmpeg.exe',
        'ffprobe_location': 'C:/path_programs_fmpeg/ffprobe.exe'
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download(['ytsearch:' + video_name])
            # Add the file path to the dictionary
            file_paths[video_name] = file_path
            return file_path
        except:
            return None

# Load the CSV file into a pandas dataframe
df = pd.read_csv('hot_100_data.csv')

# Download the audio for each song in the dataframe
df['file_path'] = df.apply(download_audio, axis=1)

# Save the updated dataframe to a new CSV file
df.to_csv('songs_with_file_paths.csv', index=False)