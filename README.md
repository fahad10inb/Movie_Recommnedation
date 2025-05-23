# Movie Recommendation System

## Project Description
This project is a Movie Recommendation System that utilizes machine learning algorithms to suggest movies based on user preferences. It features a Flask backend for handling API requests and a React frontend for user interaction.

## Features
- Search for movies by title
- Get movie recommendations based on user-selected movies
- AI agent functionality for conversational recommendations
- Health check API to ensure the service is running

## Technologies Used
- **Frontend**: React, CSS
- **Backend**: Flask, Python
- **Machine Learning**: Scikit-learn, Pandas, NumPy
- **Data Storage**: Pickle, BZ2 for model storage

## Installation Instructions
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install backend dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install frontend dependencies:
   ```bash
   cd frontend
   npm install
   ```

## Usage Instructions
1. Start the backend server:
   ```bash
   python backend/app.py
   ```
2. Start the frontend application:
   ```bash
   cd frontend
   npm start
   ```
3. Open your browser and navigate to `http://localhost:3000` to access the application.

## File Structure
```
.
├── backend
│   ├── app.py                # Flask application and API endpoints
│   ├── model.py              # Movie recommendation model
├── frontend
│   ├── src
│   │   ├── components        # React components for UI
│   │   ├── App.js            # Main application component
│   │   ├── config.js         # API configuration
│   ├── public
│   │   ├── index.html        # Main HTML file
├── data                      # Data files for model training
├── models                    # Trained model files
├── preprocess.py             # Data preprocessing script
├── README.md                 # Project documentation
```

## API Documentation
- **GET /api/health**: Check if the service is running.
- **GET /api/movies**: Retrieve a list of movies and genres.
- **GET /api/search?q={query}**: Search for movies by title.
- **POST /api/recommend**: Get movie recommendations based on a selected movie.
- **POST /api/agent**: Get recommendations based on multiple liked movies.


## License
This project is licensed under the MIT License.
