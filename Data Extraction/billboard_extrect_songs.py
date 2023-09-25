import pandas as pd
import billboard

data = {'title': [], 'artist': [], 'rank': [], 'year': []}
for y in range(2018, 2023):
    char = billboard.ChartData('hot-100-songs', year=y)
    for entry in char:
        data['title'].append(entry.title)
        data['artist'].append(entry.artist)
        data['rank'].append(entry.rank)
        data['year'].append(y)

df = pd.DataFrame(data)

# Create a set of unique song titles
unique_titles = set(df['title'])

# Create a dictionary to store the popularity of each song
popularity_dict = {}
for title in unique_titles:
    years = df.loc[df['title'] == title, 'year']
    if len(years) > 1 and 2022 in years.values:
        popularity_dict[title] = 'Popular'
    else:
        popularity_dict[title] = 'Not Popular'

# Map the popularity values to the DataFrame
df['popularity'] = df['title'].map(popularity_dict)

# Save the DataFrame to a CSV file
df.to_csv('hot_100_data.csv', index=False)