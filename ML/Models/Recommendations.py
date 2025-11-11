import pandas as pd
import numpy as np

from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.snowball import SnowballStemmer
import pickle


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

def filter_keywords(x, keywords):
    words = []
    for i in x:
        if i in keywords:
            words.append(i)
    return words

def weighted_rating(df, m, c):
    v = df['vote_count']
    r = df['vote_average']

    return (v/(v+m) * r) + (m/(m+v) * c)


def make_preprocessing():
    metadata = pd.read_csv('ML/Data/Raw/movies_metadata.csv')
    credits = pd.read_csv('ML/Data/Raw/credits.csv')
    keywords = pd.read_csv('ML/Data/Raw/keywords.csv')
    links_small = pd.read_csv('ML/Data/Raw/DataSets/Small/links_small.csv')

    metadata['genres'] = metadata['genres'].fillna('[]').apply(literal_eval).apply(
        lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    # Данные с пропусками
    metadata = metadata.drop([19730, 29503, 35587])
    metadata['id'] = metadata['id'].astype('int')

    keywords['id'] = keywords['id'].astype('int')
    credits['id'] = credits['id'].astype('int')

    links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')

    df = metadata[metadata['id'].isin(links_small)]
    df = df.merge(credits, on='id')
    df = df.merge(keywords, on='id')

    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').apply(
        lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
    df['cast'] = df['cast'].apply(literal_eval)
    df['crew'] = df['crew'].apply(literal_eval)
    df['keywords'] = df['keywords'].apply(literal_eval)
    df['cast_size'] = df['cast'].apply(lambda x: len(x))
    df['crew_size'] = df['crew'].apply(lambda x: len(x))
    df['tagline'] = df['tagline'].fillna('')
    df['description'] = df['overview'] + df['tagline']
    df['description'] = df['description'].fillna('')
    df['director'] = df['crew'].apply(get_director)
    df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    df['cast'] = df['cast'].apply(lambda x: x[:3] if len(x) >= 3 else x)
    df['keywords'] = df['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    df['cast'] = df['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    df['director'] = df['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
    df['director'] = df['director'].apply(lambda x: [x, x, x])

    keywords = df.apply(lambda x: pd.Series(x['keywords']), axis=1).stack().reset_index(level=1, drop=True)
    keywords.name = 'keyword'
    keywords = keywords.value_counts()
    keywords = keywords[keywords > 1]

    stemmer = SnowballStemmer('english')
    df['keywords'] = df['keywords'].apply(lambda x: filter_keywords(x, keywords))
    df['keywords'] = df['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
    df['keywords'] = df['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
    df['soup'] = df['keywords'] + df['cast'] + df['director'] + df['genres']
    df['soup'] = df['soup'].apply(lambda x: ' '.join(x))

    count = CountVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, stop_words='english')
    count_matrix = count.fit_transform(df['soup'])
    cosine_sim = cosine_similarity(count_matrix, dense_output=True)

    return df, cosine_sim

def get_recommendations(title, df, cosine_sim):
    df = df.reset_index()
    indices = pd.Series(df.index, index=df['title'])

    idx = indices[title]
    sim_scores = [(i, float(score)) for i, score in enumerate(cosine_sim[idx].flatten())]
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]

    valid_indices = [index for index in movie_indices if index < len(df)]
    movies = df.iloc[valid_indices][['title', 'vote_count', 'vote_average', 'year']]

    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    c = vote_averages.mean()
    m = vote_counts.quantile(0.60)

    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified = qualified.copy()
    qualified.loc[:, 'vote_count'] = qualified['vote_count'].astype('int')
    qualified.loc[:, 'vote_average'] = qualified['vote_average'].astype('int')
    qualified.loc[:, 'wr'] = qualified.apply(lambda x: weighted_rating(x, m, c), axis=1)
    qualified = qualified.sort_values('wr', ascending=False)['title'].head(5)
    return qualified


if __name__ == '__main__':
    data, cosine_similarity = make_preprocessing()

    pickle.dump(data, open('ML/Data/Processed/movie_list.pkl', 'wb'))
    pickle.dump(cosine_similarity, open('ML/Data/Processed/cosine_sim.pkl', 'wb'))