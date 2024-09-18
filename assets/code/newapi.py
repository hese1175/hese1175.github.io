import requests
import re
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

### Connecting to NewsAPI and getting data

end = 'https://newsapi.org/v2/everything'

all_jsons = []

for page in range(1, 6):
    URL_id = {'apiKey': 'de147731dffe48aa9effd3380691833b',
              'q': 'wind energy AND (recycle OR recycling OR recyclable OR recyclability)',
              'from': '1990-01-01',
              'to': datetime.now().strftime('%Y-%m-%d'),
              'language': 'en',
              'pageSize': 100,
              'page': page
              }

    response = requests.get(end, URL_id)
    jsontxt = response.json()

    all_jsons.append(jsontxt)

# Create a DataFrame from the articles
df = pd.DataFrame(all_articles)

# Convert the publishedAt column to datetime
df['publishedAt'] = pd.to_datetime(df['publishedAt'])

# Extract the year from the publishedAt column
df['year'] = df['publishedAt'].dt.year

# Group by year and count the number of articles
trends = df.groupby('year').size().reset_index(name='article_count')

# Plot the trends
fig, ax = plt.subplots()

ax.plot(trends['year'], trends['article_count'], marker='o')
ax.set_title('Trends in Wind Energy and Recyclability Articles (1990-Present)')
ax.set_xlabel('Year')
ax.set_ylabel('Number of Articles')