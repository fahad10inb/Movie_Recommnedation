import pandas as pd
from scipy.sparse import csr_matrix
import pickle

# Load and sample ratings (15M rows ~462 MB)
ratings = pd.read_csv('data/ml-25m/ratings.csv')
ratings_sample = ratings.sample(n=15000000, random_state=42)  # ~462 MB
ratings_sample.to_csv('data/ratings_sample.csv', index=False)

# Load movies and filter to match sampled ratings
movies = pd.read_csv('data/ml-25m/movies.csv')
movies_filtered = movies[movies['movieId'].isin(ratings_sample['movieId'].unique())]
movies_filtered['title'] = movies_filtered['title'].str.replace(r'\(\d+\)', '', regex=True).str.strip()
movies_filtered['genres'] = movies_filtered['genres'].str.split('|')
movies_filtered = movies_filtered.dropna()

# Compute average ratings
avg_ratings = ratings_sample.groupby('movieId')['rating'].mean().reset_index()
avg_ratings.columns = ['movieId', 'avg_rating']
movies_filtered = movies_filtered.merge(avg_ratings, on='movieId', how='left').fillna(3.0)

# Create sparse user-movie matrix (top 500 movies for efficiency)
movie_counts = ratings_sample['movieId'].value_counts()
top_movies = movie_counts.head(500).index
ratings_top = ratings_sample[ratings_sample['movieId'].isin(top_movies)]
user_ids = ratings_top['userId'].astype('category').cat.codes
movie_ids = ratings_top['movieId'].astype('category').cat.codes
movie_id_map = dict(enumerate(ratings_top['movieId'].astype('category').cat.categories))
user_movie_matrix = csr_matrix((ratings_top['rating'], (user_ids, movie_ids)),
                               shape=(user_ids.max() + 1, movie_ids.max() + 1))

# Filter movies to top 500
movies_top = movies_filtered[movies_filtered['movieId'].isin(top_movies)]

# Save processed data
movies_top.to_csv('data/clean_movies.csv', index=False)
with open('data/user_movie_matrix.pkl', 'wb') as f:
    pickle.dump(user_movie_matrix, f)
with open('data/movie_id_map.pkl', 'wb') as f:
    pickle.dump(movie_id_map, f)

print("Data processed!")
print(f"ratings_sample.csv size: {ratings_sample.memory_usage().sum() / 1024**2:.2f} MB")
print(f"movies_filtered.csv size: {movies_filtered.memory_usage().sum() / 1024**2:.2f} MB")