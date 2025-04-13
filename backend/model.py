import pandas as pd
import numpy as np
import pickle
import bz2
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from difflib import get_close_matches
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(data_path="../data/"):
    logger.info("Loading data files...")
    try:
        with open(os.path.join(data_path, 'user_movie_matrix.pkl'), 'rb') as f:
            user_movie_matrix = pickle.load(f)
        
        with open(os.path.join(data_path, 'movie_id_map.pkl'), 'rb') as f:
            movie_id_map = pickle.load(f)
        
        movies = pd.read_csv(os.path.join(data_path, 'clean_movies.csv'))
        
        logger.info(f"Loaded user-movie matrix with shape: {user_movie_matrix.shape}")
        logger.info(f"Loaded movie mapping with {len(movie_id_map)} movies")
        logger.info(f"Loaded movie details with {len(movies)} entries")
        
        return user_movie_matrix, movie_id_map, movies
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def reduce_dimensions(matrix, n_components=150):
    logger.info(f"Reducing dimensions to {n_components} components...")
    if not isinstance(matrix, csr_matrix):
        matrix = csr_matrix(matrix)
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_matrix = svd.fit_transform(matrix)
    
    variance_ratio = svd.explained_variance_ratio_.sum()
    logger.info(f"Explained variance with {n_components} components: {variance_ratio:.4f}")
    
    return reduced_matrix

def compute_chunk_similarities(chunk_start, chunk_size, features):
    chunk_end = min(chunk_start + chunk_size, features.shape[0])
    chunk = features[chunk_start:chunk_end]
    chunk_similarities = cosine_similarity(chunk, features)
    return chunk_start, chunk_similarities

def get_top_n_similar(similarity_matrix, movie_id_map, n=20):
    logger.info(f"Extracting top {n} similar movies for each movie...")
    top_n = {}
    for i in range(similarity_matrix.shape[0]):
        similar_indices = np.argpartition(similarity_matrix[i], -(n+1))[-(n+1):]
        similar_indices = similar_indices[np.argsort(-similarity_matrix[i][similar_indices])]
        similar_indices = similar_indices[1:n+1]  # exclude itself
        similar_scores = similarity_matrix[i][similar_indices]
        movie_id = movie_id_map[i]
        top_n[movie_id] = {
            movie_id_map[idx]: float(score) for idx, score in zip(similar_indices, similar_scores)
        }
    return top_n

# ✅ AI-Like Explanation Generator
def generate_justification(selected_title, recommended_title, similarity_score, common_genres):
    explanation = f"If you enjoyed '{selected_title}', you might like '{recommended_title}' "
    if common_genres:
        genre_text = ', '.join(common_genres)
        explanation += f"because it shares themes like {genre_text}. "
    explanation += f"The recommendation is based on user behavior and has a similarity score of {similarity_score:.2f}."
    return explanation

# ✅ Enhanced Recommender Function
def recommend_movies(movie_id, top_similarities, movies_df, genre=None, n=5):
    if movie_id not in top_similarities:
        return []

    similar_movies = top_similarities[movie_id]
    similar_movie_ids = list(similar_movies.keys())
    
    selected_movie_title = movies_df[movies_df['movieId'] == movie_id]['title'].values[0]
    selected_movie_genres = movies_df[movies_df['movieId'] == movie_id]['genres'].values[0]
    selected_genres = set(selected_movie_genres.split('|')) if isinstance(selected_movie_genres, str) else set()

    recs = movies_df[movies_df['movieId'].isin(similar_movie_ids)].copy()
    recs['similarity'] = recs['movieId'].apply(lambda x: similar_movies.get(x, 0))
    recs = recs.sort_values('similarity', ascending=False)

    if genre:
        recs = recs[recs['genres'].apply(lambda x: genre in eval(x) if isinstance(x, str) else False)]

    recommendations = []
    for _, row in recs.head(n).iterrows():
        rec_genres = set(row['genres'].split('|')) if isinstance(row['genres'], str) else set()
        common_genres = list(selected_genres.intersection(rec_genres))
        recommendations.append({
            'movieId': row['movieId'],
            'title': row['title'],
            'genres': list(rec_genres),
            'avg_rating': row.get('avg_rating', None),
            'similarity': row['similarity'],
            'explanation': generate_justification(
                selected_movie_title, row['title'], row['similarity'], common_genres
            )
        })

    return recommendations

# ✅ Movie Title Fuzzy Matching Helper
def get_closest_movie_titles(query, movies_df, n=3, cutoff=0.6):
    titles = movies_df['title'].tolist()
    matches = get_close_matches(query, titles, n=n, cutoff=cutoff)
    matched_movies = movies_df[movies_df['title'].isin(matches)]
    return matched_movies[['movieId', 'title']].to_dict(orient='records')

# ✅ Training Pipeline
def main():
    data_path = "../data/"
    output_path = "../models/"
    os.makedirs(output_path, exist_ok=True)
    
    user_movie_matrix, movie_id_map, movies = load_data(data_path)
    inv_movie_map = {i: movie_id for i, movie_id in movie_id_map.items()}
    
    if not isinstance(user_movie_matrix, csr_matrix):
        user_movie_matrix = csr_matrix(user_movie_matrix)
    
    reduced_features = reduce_dimensions(user_movie_matrix.T)
    
    chunk_size = 500
    n_chunks = (reduced_features.shape[0] + chunk_size - 1) // chunk_size
    logger.info(f"Computing similarities in {n_chunks} chunks...")

    results = Parallel(n_jobs=-1)(
        delayed(compute_chunk_similarities)(i * chunk_size, chunk_size, reduced_features)
        for i in range(n_chunks)
    )

    similarity_matrix = np.zeros((reduced_features.shape[0], reduced_features.shape[0]))
    for start_idx, chunk_similarities in results:
        chunk_end = min(start_idx + chunk_size, reduced_features.shape[0])
        similarity_matrix[start_idx:chunk_end] = chunk_similarities

    top_n_similarities = get_top_n_similar(similarity_matrix, inv_movie_map, n=30)

    logger.info("Saving model...")
    model_data = {
        'top_similarities': top_n_similarities,
        'movies': movies[['movieId', 'title', 'genres', 'avg_rating']]
    }

    with bz2.BZ2File(os.path.join(output_path, 'movie_recommender.pbz2'), 'wb') as f:
        pickle.dump(model_data, f)

    with open(os.path.join(output_path, 'top_similarities.pkl'), 'wb') as f:
        pickle.dump(top_n_similarities, f)

    logger.info("Model trained and saved!")

    # Sample test
    test_movie_id = list(movie_id_map.values())[0]
    test_recommendations = recommend_movies(test_movie_id, top_n_similarities, movies, n=3)
    logger.info(f"Sample AI recommendations for movie ID {test_movie_id}:")
    for rec in test_recommendations:
        logger.info(f"  - {rec['title']} → {rec['explanation']}")

if __name__ == "__main__":
    main()
