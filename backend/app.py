"""
Movie Recommendation AI Agent
Complete implementation including model training, Flask API, and agent functionality
"""
import os
import logging
import pickle
import bz2
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
from difflib import get_close_matches

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_PATH = "../data/"
MODEL_PATH = "../models/"
os.makedirs(MODEL_PATH, exist_ok=True)

# =====================================
# MODEL TRAINING AND CORE FUNCTIONS
# =====================================

def load_data(data_path=DATA_PATH):
    """Load data files for model training"""
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
    """Reduce dimensionality of user-movie matrix using SVD"""
    logger.info(f"Reducing dimensions to {n_components} components...")
    if not isinstance(matrix, csr_matrix):
        matrix = csr_matrix(matrix)
    
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_matrix = svd.fit_transform(matrix)
    
    variance_ratio = svd.explained_variance_ratio_.sum()
    logger.info(f"Explained variance with {n_components} components: {variance_ratio:.4f}")
    
    return reduced_matrix

def compute_chunk_similarities(chunk_start, chunk_size, features):
    """Compute cosine similarities for a chunk of the matrix"""
    chunk_end = min(chunk_start + chunk_size, features.shape[0])
    chunk = features[chunk_start:chunk_end]
    chunk_similarities = cosine_similarity(chunk, features)
    return chunk_start, chunk_similarities

def get_top_n_similar(similarity_matrix, movie_id_map, n=20):
    """Extract top N similar movies for each movie"""
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

def generate_justification(selected_title, recommended_title, similarity_score, common_genres):
    """Generate natural language explanation for recommendation"""
    explanation = f"If you enjoyed '{selected_title}', you might like '{recommended_title}' "
    if common_genres:
        genre_text = ', '.join(common_genres)
        explanation += f"because it shares themes like {genre_text}. "
    explanation += f"The recommendation is based on user behavior and has a similarity score of {similarity_score:.2f}."
    return explanation

def generate_agent_response(liked_movies, recommendations, genre=None):
    """Generate conversational agent response"""
    if not recommendations:
        return "I couldn't find any recommendations that match your preferences."
    
    if len(liked_movies) == 1:
        response = f"Based on your interest in '{liked_movies[0]}', "
    else:
        movie_list = "', '".join(liked_movies[:-1])
        response = f"Based on your interest in '{movie_list}' and '{liked_movies[-1]}', "
    
    if genre:
        response += f"and your preference for {genre} films, "
    
    response += f"here are {len(recommendations)} movies you might enjoy. "
    
    # Add highlight for top recommendation
    top_rec = recommendations[0]
    response += f"I especially recommend '{top_rec['title']}' - {top_rec['explanation'].split(' - ')[1] if ' - ' in top_rec['explanation'] else top_rec['explanation']}"
    
    return response

def get_closest_movie_titles(query, movies_df, n=3, cutoff=0.6):
    """Find closest movie titles using fuzzy matching with preference for exact matches"""
    # Convert query to lowercase for case-insensitive comparison
    query_lower = query.lower().strip()
    
    # Step 1: Try exact match first (case-insensitive)
    exact_matches = movies_df[movies_df['title'].str.lower() == query_lower]
    if not exact_matches.empty:
        logger.info(f"Exact match found for '{query}': {exact_matches.iloc[0]['title']}")
        return exact_matches[['movieId', 'title']].to_dict(orient='records')
    
    # Step 2: Try contains match (for partial titles)
    contains_matches = movies_df[movies_df['title'].str.lower().str.contains(query_lower)]
    if not contains_matches.empty and len(query_lower) > 3:  # Only use for queries of reasonable length
        # Sort by title length (prefer shorter titles as they're more likely to be exact)
        contains_matches = contains_matches.assign(title_len=contains_matches['title'].str.len())
        contains_matches = contains_matches.sort_values('title_len')
        logger.info(f"Contains match found for '{query}': {contains_matches.iloc[0]['title']}")
        return contains_matches[['movieId', 'title']].head(n).to_dict(orient='records')
    
    # Step 3: Fall back to fuzzy matching with higher cutoff
    titles = movies_df['title'].tolist()
    matches = get_close_matches(query, titles, n=n, cutoff=max(cutoff, 0.7))  # Increased cutoff
    
    if matches:
        matched_movies = movies_df[movies_df['title'].isin(matches)]
        logger.info(f"Fuzzy match found for '{query}': {matched_movies.iloc[0]['title']} (score: ≈{cutoff})")
        return matched_movies[['movieId', 'title']].to_dict(orient='records')
    
    # Step 4: If all else fails, try one more time with lower cutoff
    matches = get_close_matches(query, titles, n=n, cutoff=cutoff)
    matched_movies = movies_df[movies_df['title'].isin(matches)]
    
    if not matched_movies.empty:
        logger.info(f"Loose fuzzy match found for '{query}': {matched_movies.iloc[0]['title']} (score: ≈{cutoff})")
        return matched_movies[['movieId', 'title']].to_dict(orient='records')
    
    logger.info(f"No match found for '{query}'")
    return []

def recommend_movies(movie_id, top_similarities, movies_df, genre=None, n=5):
    """Generate movie recommendations based on a given movie ID"""
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

    if genre and not recs.empty:
        recs = recs[recs['genres'].apply(lambda x: genre in x.split('|') if isinstance(x, str) else False)]

    recommendations = []
    for _, row in recs.head(n).iterrows():
        rec_genres = set(row['genres'].split('|')) if isinstance(row['genres'], str) else set()
        common_genres = list(selected_genres.intersection(rec_genres))
        recommendations.append({
            'movieId': row['movieId'],
            'title': row['title'],
            'genres': list(rec_genres),
            'avg_rating': row.get('avg_rating', None),
            'similarity': float(row['similarity']),
            'explanation': generate_justification(
                selected_movie_title, row['title'], row['similarity'], common_genres
            )
        })

    return recommendations

def train_model():
    """Train the recommendation model and save it"""
    user_movie_matrix, movie_id_map, movies = load_data()
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

    with bz2.BZ2File(os.path.join(MODEL_PATH, 'movie_recommender.pbz2'), 'wb') as f:
        pickle.dump(model_data, f)

    logger.info("Model trained and saved!")
    return top_n_similarities, movies

def load_model():
    """Load the trained model"""
    try:
        logger.info("Loading model from file...")
        with bz2.BZ2File(os.path.join(MODEL_PATH, 'movie_recommender.pbz2'), 'rb') as f:
            model_data = pickle.load(f)
        return model_data['top_similarities'], model_data['movies']
    except FileNotFoundError:
        logger.info("Model file not found, training new model...")
        return train_model()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# =====================================
# FLASK API IMPLEMENTATION
# =====================================

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global storage for user sessions
user_sessions = {}

# Load model at startup
top_similarities, movies_df = load_model()

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "model_loaded": top_similarities is not None}), 200

@app.route('/api/movies', methods=['GET'])
def get_movies():
    """Get list of movies and genres"""
    # Extract unique genres from the dataset
    all_genres = []
    for genres in movies_df['genres'].dropna():
        if isinstance(genres, str):
            all_genres.extend(genres.split('|'))
    unique_genres = sorted(list(set(all_genres)))
    
    return jsonify({
        'movies': movies_df['title'].tolist(),
        'genres': unique_genres
    })

@app.route('/api/search', methods=['GET'])
def search_movies():
    """Search for movies by title"""
    query = request.args.get('q', '')
    if not query or len(query) < 3:
        return jsonify({'results': []}), 400
    
    matches = get_closest_movie_titles(query, movies_df, n=5, cutoff=0.5)
    return jsonify({'results': matches})

@app.route('/api/recommend', methods=['POST'])
def recommend():
    """Generate recommendations based on a single movie"""
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON data'}), 400
        
    movie_title = data.get('movie')
    genre = data.get('genre')
    num_recommendations = data.get('num', 5)
    
    if not movie_title:
        return jsonify({'error': 'Movie title required'}), 400
    
    logger.info(f"Recommendation requested for: '{movie_title}', genre: '{genre}'")
    matches = get_closest_movie_titles(movie_title, movies_df)
    
    if not matches:
        return jsonify({'error': f'No match found for "{movie_title}"'}), 404
    
    # Get the matched movie
    matched_movie = matches[0]
    movie_id = matched_movie['movieId']
    
    # Check if match is significantly different from request
    exact_match = matched_movie['title'].lower() == movie_title.lower()
    if not exact_match and len(movie_title) > 3:
        # Log the discrepancy
        logger.warning(f"User requested '{movie_title}' but matched to '{matched_movie['title']}'")
    
    recommendations = recommend_movies(
        movie_id, 
        top_similarities, 
        movies_df, 
        genre, 
        n=num_recommendations
    )
    
    if not recommendations:
        logger.warning(f"No recommendations found for movie ID {movie_id} ({matched_movie['title']})")
        
    return jsonify({
        'recommendations': recommendations, 
        'selected_movie': matched_movie['title'],
        'original_query': movie_title,
        'exact_match': exact_match
    })

@app.route('/api/agent', methods=['POST'])
def ai_agent():
    """AI agent endpoint that handles multiple liked movies"""
    data = request.get_json()
    liked_movies = data.get('liked_movies', [])
    genre = data.get('genre')
    session_id = data.get('session_id')
    
    logger.info(f"Agent recommendations requested for: {liked_movies}, genre: {genre}")
    
    # Store in session if session_id provided
    if session_id:
        user_sessions[session_id] = user_sessions.get(session_id, {})
        user_sessions[session_id]['liked_movies'] = liked_movies
        user_sessions[session_id]['genre'] = genre
    
    movie_ids = []
    matched_titles = []
    for title in liked_movies:
        match = get_closest_movie_titles(title, movies_df)
        if match:
            movie_ids.append(match[0]['movieId'])
            matched_titles.append(match[0]['title'])
            if match[0]['title'].lower() != title.lower():
                logger.info(f"Matched '{title}' to '{match[0]['title']}'")
    
    if not movie_ids:
        return jsonify({'error': 'No valid movies found'}), 400
    
    all_recs = []
    for mid in movie_ids:
        recs = recommend_movies(mid, top_similarities, movies_df, genre, n=10)
        all_recs.extend(recs)
    
    # Combine recommendations and rank by similarity
    if all_recs:
        all_recs_df = pd.DataFrame(all_recs).drop_duplicates('movieId')
        all_recs_df = all_recs_df.sort_values('similarity', ascending=False)
        final_recs = all_recs_df.head(7).to_dict(orient='records')
    else:
        final_recs = []
        logger.warning(f"No recommendations found for movies: {matched_titles}")
    
    # Generate agent response
    agent_response = generate_agent_response(matched_titles, final_recs, genre)
    
    return jsonify({
        'agent_reply': agent_response,
        'recommendations': final_recs,
        'matched_titles': matched_titles,  # Return what the movie titles were matched to
        'original_titles': liked_movies    # Return the original titles for comparison
    })

@app.route('/api/agent/session', methods=['POST'])
def agent_session():
    """Advanced agent endpoint with conversational context"""
    data = request.get_json()
    session_id = data.get('session_id')
    message = data.get('message', '')
    
    logger.info(f"Session message received: '{message}' for session {session_id}")
    
    # Create or retrieve session
    if session_id not in user_sessions:
        user_sessions[session_id] = {
            'liked_movies': [],
            'disliked_movies': [],
            'preferred_genres': [],
            'conversation_history': []
        }
    
    session = user_sessions[session_id]
    session['conversation_history'].append({'user': message})
    
    # Simple NLP to extract movie mentions and sentiment
    # In a real system, you'd use a proper NLP model here
    potential_movies = get_movie_mentions(message, movies_df)
    
    # Check for genre mentions
    genres = extract_genres(message, movies_df)
    if genres:
        session['preferred_genres'] = genres
    
    # Determine if the message has positive or negative sentiment
    # Again, in a real system use a proper sentiment analyzer
    is_negative = any(word in message.lower() for word in ['dislike', 'hate', 'boring', 'bad', 'terrible'])
    
    # Add extracted movies to liked/disliked lists    
    for movie in potential_movies:
        if is_negative:
            if movie['title'] not in session['disliked_movies']:
                session['disliked_movies'].append(movie['title'])
        else:
            if movie['title'] not in session['liked_movies']:
                session['liked_movies'].append(movie['title'])
    
    # Generate recommendations based on session context
    recommendations = generate_session_recommendations(session, top_similarities, movies_df)
    
    # Generate conversational response
    if not potential_movies and not genres:
        reply = "I'm not sure which movies you're referring to. Could you mention some movies you've enjoyed?"
    elif not recommendations:
        reply = "I couldn't find recommendations matching your preferences. Could you mention some other movies you like?"
    else:
        if is_negative:
            reply = f"I'll avoid recommending movies like {', '.join([m['title'] for m in potential_movies])}. "
        else:
            movie_list = ', '.join([f"'{m['title']}'" for m in potential_movies])
            reply = f"Based on your interest in {movie_list}"
            if genres:
                reply += f" and preference for {', '.join(genres)} films"
            reply += ", here are some recommendations you might enjoy:"
    
    session['conversation_history'].append({'agent': reply})
    
    return jsonify({
        'agent_reply': reply,
        'recommendations': recommendations,
        'extracted_movies': [m['title'] for m in potential_movies],
        'extracted_genres': genres
    })

def get_movie_mentions(message, movies_df):
    """Extract movie mentions from user message"""
    # This is a simple approach - in a production system use NER
    results = []
    # Split into potential title fragments
    words = message.split()
    for i in range(len(words)):
        for j in range(i+1, min(i+8, len(words)+1)):  # max 7 word titles
            potential_title = ' '.join(words[i:j])
            matches = get_closest_movie_titles(potential_title, movies_df, cutoff=0.8)
            if matches:
                results.append(matches[0])
    
    # Remove duplicates
    unique_results = []
    seen_ids = set()
    for match in results:
        if match['movieId'] not in seen_ids:
            seen_ids.add(match['movieId'])
            unique_results.append(match)
    
    return unique_results

def extract_genres(message, movies_df):
    """Extract genre mentions from user message"""
    # Get all unique genres from dataset
    all_genres = []
    for genres in movies_df['genres'].dropna():
        if isinstance(genres, str):
            all_genres.extend(genres.split('|'))
    unique_genres = list(set(all_genres))
    
    found_genres = []
    for genre in unique_genres:
        if genre.lower() in message.lower():
            found_genres.append(genre)
    
    return found_genres

def generate_session_recommendations(session, top_similarities, movies_df, n=5):
    """Generate recommendations based on session context"""
    liked_movie_ids = []
    
    # Convert liked movies to IDs
    for title in session.get('liked_movies', []):
        match = get_closest_movie_titles(title, movies_df)
        if match:
            liked_movie_ids.append(match[0]['movieId'])
    
    if not liked_movie_ids:
        return []
    
    # Get recommendations for each liked movie
    all_recs = []
    for mid in liked_movie_ids:
        genre = session.get('preferred_genres', [None])[0]
        recs = recommend_movies(mid, top_similarities, movies_df, genre, n=10)
        all_recs.extend(recs)
    
    # Remove any disliked movies
    disliked_titles = session.get('disliked_movies', [])
    filtered_recs = [rec for rec in all_recs if rec['title'] not in disliked_titles]
    
    # Combine and rank
    if filtered_recs:
        recs_df = pd.DataFrame(filtered_recs).drop_duplicates('movieId')
        recs_df = recs_df.sort_values('similarity', ascending=False)
        return recs_df.head(n).to_dict(orient='records')
    
    return []

# =====================================
# MAIN ENTRY POINT
# =====================================

if __name__ == "__main__":
    # Check if model exists, if not train it
    if not os.path.exists(os.path.join(MODEL_PATH, 'movie_recommender.pbz2')):
        logger.info("Training new model...")
        train_model()
    
    # Run Flask app with improved error handling
    try:
        logger.info("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Error starting Flask server: {e}")