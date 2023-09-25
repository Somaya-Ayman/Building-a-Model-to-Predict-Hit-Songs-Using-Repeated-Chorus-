import pandas as pd
import librosa
import numpy as np
from scipy.stats import skew, kurtosis

# Load the list of songs into a pandas dataframe
df = pd.read_csv('songs_with_chorus_paths.csv')

def stat(list, feature_name, columns, data):
    count = 0
    for i in list:

        Skew = skew(i)
        columns.append(f'{feature_name}_kew_{count}')
        min = np.min(i)
        columns.append(f'{feature_name}_min_{count}')
        max = np.max(i)
        columns.append(f'{feature_name}_max_{count}')
        std = np.std(i)
        columns.append(f'{feature_name}_std_{count}')
        mean = np.mean(i)
        columns.append(f'{feature_name}_mean_{count}')
        median = np.median(i)
        columns.append(f'{feature_name}_median_{count}')
        Kurtosis = kurtosis(i)
        columns.append(f'{feature_name}_kurtosis_{count}')

        count += 1

        data.append(Skew)
        data.append(min)
        data.append(max)
        data.append(std)
        data.append(mean)
        data.append(median)
        data.append(Kurtosis)

    return data


# Define a function to extract the mean chroma features of the chorus for a given song
def extract_features(audio_path, title):
    data = []
    columns_name = ['title']
    data.append(title)

    try:
        # load audio waveform and sample rate
        audio, sampling_rate = librosa.load(audio_path)

        chroma_stft = librosa.feature.chroma_stft(y=audio, sr=sampling_rate)
        stat(chroma_stft, 'chroma_stft', columns_name, data)

        chroma_cqt = librosa.feature.chroma_cqt(y=audio, sr=sampling_rate)
        stat(chroma_cqt, 'chroma_cqt', columns_name, data)

        chroma_cens = librosa.feature.chroma_cens(y=audio, sr=sampling_rate)
        stat(chroma_cens, 'chroma_cens', columns_name, data)

        mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate)
        stat(mfcc, 'mfcc', columns_name, data)

        rms = librosa.feature.rms(y=audio)
        stat(rms, 'rms', columns_name, data)

        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sampling_rate)
        stat(spectral_centroid, 'spectral_centroid', columns_name, data)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sampling_rate)
        stat(spectral_bandwidth, 'spectral_bandwidth', columns_name, data)

        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sampling_rate)
        stat(spectral_contrast, 'spectral_contrast', columns_name, data)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sampling_rate)
        stat(spectral_rolloff, 'spectral_rolloff', columns_name, data)

        tonnetz = librosa.feature.tonnetz(y=audio, sr=sampling_rate)
        stat(tonnetz, 'tonnetz', columns_name, data)

        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
        stat(zero_crossing_rate, 'zero_crossing_rate', columns_name, data)

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None, None

    if data:
        print("d:", data)
        print("cols:", columns_name)
        return data, columns_name
    else:
        return None, None

i = 0
data = []
for i in range(len(df.chorus_path)):
    audio_path = "C:/Users/lenovo/Documents/gitt/major project technolabs 1/Building-a-Model-to-Predict-Hit-Songs-Using-Repeated-Chorus/" + str(df.chorus_path[i])
    d, cols = extract_features(audio_path, df.title[i])
    if d is not None and cols is not None:
        data.append(d)
    print(f'The {i} song Done...')

# Define column names for the DataFrame
column_names = ['title', 'chroma_stft_kew_0', 'chroma_stft_min_0', 'chroma_stft_max_0', 'chroma_stft_std_0', 'chroma_stft_mean_0', 'chroma_stft_median_0', 'chroma_stft_kurtosis_0', 'chroma_stft_kew_1', 'chroma_stft_min_1', 'chroma_stft_max_1', 'chroma_stft_std_1', 'chroma_stft_mean_1', 'chroma_stft_median_1', 'chroma_stft_kurtosis_1', 'chroma_stft_kew_2', 'chroma_stft_min_2', 'chroma_stft_max_2', 'chroma_stft_std_2', 'chroma_stft_mean_2', 'chroma_stft_median_2', 'chroma_stft_kurtosis_2', 'chroma_stft_kew_3', 'chroma_stft_min_3', 'chroma_stft_max_3', 'chroma_stft_std_3', 'chroma_stft_mean_3', 'chroma_stft_median_3', 'chroma_stft_kurtosis_3', 'chroma_stft_kew_4', 'chroma_stft_min_4', 'chroma_stft_max_4', 'chroma_stft_std_4', 'chroma_stft_mean_4', 'chroma_stft_median_4', 'chroma_stft_kurtosis_4', 'chroma_stft_kew_5', 'chroma_stft_min_5', 'chroma_stft_max_5', 'chroma_stft_std_5', 'chroma_stft_mean_5', 'chroma_stft_median_5', 'chroma_stft_kurtosis_5', 'chroma_stft_kew_6', 'chroma_stft_min_6', 'chroma_stft_max_6', 'chroma_stft_std_6', 'chroma_stft_mean_6', 'chroma_stft_median_6', 'chroma_stft_kurtosis_6', 'chroma_stft_kew_7', 'chroma_stft_min_7', 'chroma_stft_max_7', 'chroma_stft_std_7', 'chroma_stft_mean_7', 'chroma_stft_median_7', 'chroma_stft_kurtosis_7', 'chroma_stft_kew_8', 'chroma_stft_min_8', 'chroma_stft_max_8', 'chroma_stft_std_8', 'chroma_stft_mean_8', 'chroma_stft_median_8', 'chroma_stft_kurtosis_8', 'chroma_stft_kew_9', 'chroma_stft_min_9', 'chroma_stft_max_9', 'chroma_stft_std_9', 'chroma_stft_mean_9', 'chroma_stft_median_9', 'chroma_stft_kurtosis_9', 'chroma_stft_kew_10', 'chroma_stft_min_10', 'chroma_stft_max_10', 'chroma_stft_std_10', 'chroma_stft_mean_10', 'chroma_stft_median_10', 'chroma_stft_kurtosis_10', 'chroma_stft_kew_11', 'chroma_stft_min_11', 'chroma_stft_max_11', 'chroma_stft_std_11', 'chroma_stft_mean_11', 'chroma_stft_median_11', 'chroma_stft_kurtosis_11', 'chroma_cqt_kew_0', 'chroma_cqt_min_0', 'chroma_cqt_max_0', 'chroma_cqt_std_0', 'chroma_cqt_mean_0', 'chroma_cqt_median_0', 'chroma_cqt_kurtosis_0', 'chroma_cqt_kew_1', 'chroma_cqt_min_1', 'chroma_cqt_max_1', 'chroma_cqt_std_1', 'chroma_cqt_mean_1', 'chroma_cqt_median_1', 'chroma_cqt_kurtosis_1', 'chroma_cqt_kew_2', 'chroma_cqt_min_2', 'chroma_cqt_max_2', 'chroma_cqt_std_2', 'chroma_cqt_mean_2', 'chroma_cqt_median_2', 'chroma_cqt_kurtosis_2', 'chroma_cqt_kew_3', 'chroma_cqt_min_3', 'chroma_cqt_max_3', 'chroma_cqt_std_3', 'chroma_cqt_mean_3', 'chroma_cqt_median_3', 'chroma_cqt_kurtosis_3', 'chroma_cqt_kew_4', 'chroma_cqt_min_4', 'chroma_cqt_max_4', 'chroma_cqt_std_4', 'chroma_cqt_mean_4', 'chroma_cqt_median_4', 'chroma_cqt_kurtosis_4', 'chroma_cqt_kew_5', 'chroma_cqt_min_5', 'chroma_cqt_max_5', 'chroma_cqt_std_5', 'chroma_cqt_mean_5', 'chroma_cqt_median_5', 'chroma_cqt_kurtosis_5', 'chroma_cqt_kew_6', 'chroma_cqt_min_6', 'chroma_cqt_max_6', 'chroma_cqt_std_6', 'chroma_cqt_mean_6', 'chroma_cqt_median_6', 'chroma_cqt_kurtosis_6', 'chroma_cqt_kew_7', 'chroma_cqt_min_7', 'chroma_cqt_max_7', 'chroma_cqt_std_7', 'chroma_cqt_mean_7', 'chroma_cqt_median_7', 'chroma_cqt_kurtosis_7', 'chroma_cqt_kew_8', 'chroma_cqt_min_8', 'chroma_cqt_max_8', 'chroma_cqt_std_8', 'chroma_cqt_mean_8', 'chroma_cqt_median_8', 'chroma_cqt_kurtosis_8', 'chroma_cqt_kew_9', 'chroma_cqt_min_9', 'chroma_cqt_max_9', 'chroma_cqt_std_9', 'chroma_cqt_mean_9', 'chroma_cqt_median_9', 'chroma_cqt_kurtosis_9', 'chroma_cqt_kew_10', 'chroma_cqt_min_10', 'chroma_cqt_max_10', 'chroma_cqt_std_10', 'chroma_cqt_mean_10', 'chroma_cqt_median_10', 'chroma_cqt_kurtosis_10', 'chroma_cqt_kew_11', 'chroma_cqt_min_11', 'chroma_cqt_max_11', 'chroma_cqt_std_11', 'chroma_cqt_mean_11', 'chroma_cqt_median_11', 'chroma_cqt_kurtosis_11', 'chroma_cens_kew_0', 'chroma_cens_min_0', 'chroma_cens_max_0', 'chroma_cens_std_0', 'chroma_cens_mean_0', 'chroma_cens_median_0', 'chroma_cens_kurtosis_0', 'chroma_cens_kew_1', 'chroma_cens_min_1', 'chroma_cens_max_1', 'chroma_cens_std_1', 'chroma_cens_mean_1', 'chroma_cens_median_1', 'chroma_cens_kurtosis_1', 'chroma_cens_kew_2', 'chroma_cens_min_2', 'chroma_cens_max_2', 'chroma_cens_std_2', 'chroma_cens_mean_2', 'chroma_cens_median_2', 'chroma_cens_kurtosis_2', 'chroma_cens_kew_3', 'chroma_cens_min_3', 'chroma_cens_max_3', 'chroma_cens_std_3', 'chroma_cens_mean_3', 'chroma_cens_median_3', 'chroma_cens_kurtosis_3', 'chroma_cens_kew_4', 'chroma_cens_min_4', 'chroma_cens_max_4', 'chroma_cens_std_4', 'chroma_cens_mean_4', 'chroma_cens_median_4', 'chroma_cens_kurtosis_4', 'chroma_cens_kew_5', 'chroma_cens_min_5', 'chroma_cens_max_5', 'chroma_cens_std_5', 'chroma_cens_mean_5', 'chroma_cens_median_5', 'chroma_cens_kurtosis_5', 'chroma_cens_kew_6', 'chroma_cens_min_6', 'chroma_cens_max_6', 'chroma_cens_std_6', 'chroma_cens_mean_6', 'chroma_cens_median_6', 'chroma_cens_kurtosis_6', 'chroma_cens_kew_7', 'chroma_cens_min_7', 'chroma_cens_max_7', 'chroma_cens_std_7', 'chroma_cens_mean_7', 'chroma_cens_median_7', 'chroma_cens_kurtosis_7', 'chroma_cens_kew_8', 'chroma_cens_min_8', 'chroma_cens_max_8', 'chroma_cens_std_8', 'chroma_cens_mean_8', 'chroma_cens_median_8', 'chroma_cens_kurtosis_8', 'chroma_cens_kew_9', 'chroma_cens_min_9', 'chroma_cens_max_9', 'chroma_cens_std_9', 'chroma_cens_mean_9', 'chroma_cens_median_9', 'chroma_cens_kurtosis_9', 'chroma_cens_kew_10', 'chroma_cens_min_10', 'chroma_cens_max_10', 'chroma_cens_std_10', 'chroma_cens_mean_10', 'chroma_cens_median_10', 'chroma_cens_kurtosis_10', 'chroma_cens_kew_11', 'chroma_cens_min_11', 'chroma_cens_max_11', 'chroma_cens_std_11', 'chroma_cens_mean_11', 'chroma_cens_median_11', 'chroma_cens_kurtosis_11', 'mfcc_kew_0', 'mfcc_min_0', 'mfcc_max_0', 'mfcc_std_0', 'mfcc_mean_0', 'mfcc_median_0', 'mfcc_kurtosis_0', 'mfcc_kew_1', 'mfcc_min_1', 'mfcc_max_1', 'mfcc_std_1', 'mfcc_mean_1', 'mfcc_median_1', 'mfcc_kurtosis_1', 'mfcc_kew_2', 'mfcc_min_2', 'mfcc_max_2', 'mfcc_std_2', 'mfcc_mean_2', 'mfcc_median_2', 'mfcc_kurtosis_2', 'mfcc_kew_3', 'mfcc_min_3', 'mfcc_max_3', 'mfcc_std_3', 'mfcc_mean_3', 'mfcc_median_3', 'mfcc_kurtosis_3', 'mfcc_kew_4', 'mfcc_min_4', 'mfcc_max_4', 'mfcc_std_4', 'mfcc_mean_4', 'mfcc_median_4', 'mfcc_kurtosis_4', 'mfcc_kew_5', 'mfcc_min_5', 'mfcc_max_5', 'mfcc_std_5', 'mfcc_mean_5', 'mfcc_median_5', 'mfcc_kurtosis_5', 'mfcc_kew_6', 'mfcc_min_6', 'mfcc_max_6', 'mfcc_std_6', 'mfcc_mean_6', 'mfcc_median_6', 'mfcc_kurtosis_6', 'mfcc_kew_7', 'mfcc_min_7', 'mfcc_max_7', 'mfcc_std_7', 'mfcc_mean_7', 'mfcc_median_7', 'mfcc_kurtosis_7', 'mfcc_kew_8', 'mfcc_min_8', 'mfcc_max_8', 'mfcc_std_8', 'mfcc_mean_8', 'mfcc_median_8', 'mfcc_kurtosis_8', 'mfcc_kew_9', 'mfcc_min_9', 'mfcc_max_9', 'mfcc_std_9', 'mfcc_mean_9', 'mfcc_median_9', 'mfcc_kurtosis_9', 'mfcc_kew_10', 'mfcc_min_10', 'mfcc_max_10', 'mfcc_std_10', 'mfcc_mean_10', 'mfcc_median_10', 'mfcc_kurtosis_10', 'mfcc_kew_11', 'mfcc_min_11', 'mfcc_max_11', 'mfcc_std_11', 'mfcc_mean_11', 'mfcc_median_11', 'mfcc_kurtosis_11', 'mfcc_kew_12', 'mfcc_min_12', 'mfcc_max_12', 'mfcc_std_12', 'mfcc_mean_12', 'mfcc_median_12', 'mfcc_kurtosis_12', 'mfcc_kew_13', 'mfcc_min_13', 'mfcc_max_13', 'mfcc_std_13', 'mfcc_mean_13', 'mfcc_median_13', 'mfcc_kurtosis_13', 'mfcc_kew_14', 'mfcc_min_14', 'mfcc_max_14', 'mfcc_std_14', 'mfcc_mean_14', 'mfcc_median_14', 'mfcc_kurtosis_14', 'mfcc_kew_15', 'mfcc_min_15', 'mfcc_max_15', 'mfcc_std_15', 'mfcc_mean_15', 'mfcc_median_15', 'mfcc_kurtosis_15', 'mfcc_kew_16', 'mfcc_min_16', 'mfcc_max_16', 'mfcc_std_16', 'mfcc_mean_16', 'mfcc_median_16', 'mfcc_kurtosis_16', 'mfcc_kew_17', 'mfcc_min_17', 'mfcc_max_17', 'mfcc_std_17', 'mfcc_mean_17', 'mfcc_median_17', 'mfcc_kurtosis_17', 'mfcc_kew_18', 'mfcc_min_18', 'mfcc_max_18', 'mfcc_std_18', 'mfcc_mean_18', 'mfcc_median_18', 'mfcc_kurtosis_18', 'mfcc_kew_19', 'mfcc_min_19', 'mfcc_max_19', 'mfcc_std_19', 'mfcc_mean_19', 'mfcc_median_19', 'mfcc_kurtosis_19', 'rms_kew_0', 'rms_min_0', 'rms_max_0', 'rms_std_0', 'rms_mean_0', 'rms_median_0', 'rms_kurtosis_0', 'spectral_centroid_kew_0', 'spectral_centroid_min_0', 'spectral_centroid_max_0', 'spectral_centroid_std_0', 'spectral_centroid_mean_0', 'spectral_centroid_median_0', 'spectral_centroid_kurtosis_0', 'spectral_bandwidth_kew_0', 'spectral_bandwidth_min_0', 'spectral_bandwidth_max_0', 'spectral_bandwidth_std_0', 'spectral_bandwidth_mean_0', 'spectral_bandwidth_median_0', 'spectral_bandwidth_kurtosis_0', 'spectral_contrast_kew_0', 'spectral_contrast_min_0', 'spectral_contrast_max_0', 'spectral_contrast_std_0', 'spectral_contrast_mean_0', 'spectral_contrast_median_0', 'spectral_contrast_kurtosis_0', 'spectral_contrast_kew_1', 'spectral_contrast_min_1', 'spectral_contrast_max_1', 'spectral_contrast_std_1', 'spectral_contrast_mean_1', 'spectral_contrast_median_1', 'spectral_contrast_kurtosis_1', 'spectral_contrast_kew_2', 'spectral_contrast_min_2', 'spectral_contrast_max_2', 'spectral_contrast_std_2', 'spectral_contrast_mean_2', 'spectral_contrast_median_2', 'spectral_contrast_kurtosis_2', 'spectral_contrast_kew_3', 'spectral_contrast_min_3', 'spectral_contrast_max_3', 'spectral_contrast_std_3', 'spectral_contrast_mean_3', 'spectral_contrast_median_3', 'spectral_contrast_kurtosis_3', 'spectral_contrast_kew_4', 'spectral_contrast_min_4', 'spectral_contrast_max_4', 'spectral_contrast_std_4', 'spectral_contrast_mean_4', 'spectral_contrast_median_4', 'spectral_contrast_kurtosis_4', 'spectral_contrast_kew_5', 'spectral_contrast_min_5', 'spectral_contrast_max_5', 'spectral_contrast_std_5', 'spectral_contrast_mean_5', 'spectral_contrast_median_5', 'spectral_contrast_kurtosis_5', 'spectral_contrast_kew_6', 'spectral_contrast_min_6', 'spectral_contrast_max_6', 'spectral_contrast_std_6', 'spectral_contrast_mean_6', 'spectral_contrast_median_6', 'spectral_contrast_kurtosis_6', 'spectral_rolloff_kew_0', 'spectral_rolloff_min_0', 'spectral_rolloff_max_0', 'spectral_rolloff_std_0', 'spectral_rolloff_mean_0', 'spectral_rolloff_median_0', 'spectral_rolloff_kurtosis_0', 'tonnetz_kew_0', 'tonnetz_min_0', 'tonnetz_max_0', 'tonnetz_std_0', 'tonnetz_mean_0', 'tonnetz_median_0', 'tonnetz_kurtosis_0', 'tonnetz_kew_1', 'tonnetz_min_1', 'tonnetz_max_1', 'tonnetz_std_1', 'tonnetz_mean_1', 'tonnetz_median_1', 'tonnetz_kurtosis_1', 'tonnetz_kew_2', 'tonnetz_min_2', 'tonnetz_max_2', 'tonnetz_std_2', 'tonnetz_mean_2', 'tonnetz_median_2', 'tonnetz_kurtosis_2', 'tonnetz_kew_3', 'tonnetz_min_3', 'tonnetz_max_3', 'tonnetz_std_3', 'tonnetz_mean_3', 'tonnetz_median_3', 'tonnetz_kurtosis_3', 'tonnetz_kew_4', 'tonnetz_min_4', 'tonnetz_max_4', 'tonnetz_std_4', 'tonnetz_mean_4', 'tonnetz_median_4', 'tonnetz_kurtosis_4', 'tonnetz_kew_5', 'tonnetz_min_5', 'tonnetz_max_5', 'tonnetz_std_5', 'tonnetz_mean_5', 'tonnetz_median_5', 'tonnetz_kurtosis_5', 'zero_crossing_rate_kew_0', 'zero_crossing_rate_min_0', 'zero_crossing_rate_max_0', 'zero_crossing_rate_std_0', 'zero_crossing_rate_mean_0', 'zero_crossing_rate_median_0', 'zero_crossing_rate_kurtosis_0']  
# Convert data to pandas DataFrame with column names
dfff = pd.DataFrame(data, columns=column_names)
# Group dataframe by title and count the number of occurrences
song_counts = dfff.groupby('title').size().reset_index(name='count')

# Identify songs with repeated choruses (count > 1)
popular_songs = song_counts[song_counts['count'] > 1]['title']

# Create new column "popularity" and set value to 1 for popular songs and 0 for others
dfff['popularity'] = np.where(dfff['title'].isin(popular_songs), 1, 0)
# Write DataFrame to CSV file
dfff.to_csv('finishedd.csv', index=False)
