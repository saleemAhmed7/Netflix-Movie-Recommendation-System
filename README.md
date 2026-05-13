# Web-Based Movie Recommendation System

This is a complete university project for a Data Mining course. It uses the MovieLens dataset and the K-Nearest Neighbors (KNN) algorithm to build a web-based movie recommendation system.

## Features
- **Movie Recommendations**: Recommends top 5 or 10 movies based on:
  - Previously liked movie (KNN similarity)
  - Genre
  - Minimum rating
  - Year range
- **Dataset Analysis Dashboard**: Provides comprehensive statistics, including top-rated movies, most-rated movies, and a genre distribution chart.
- **Modern UI/UX**: Built with Bootstrap 5, featuring a responsive, cinema-inspired dark mode theme.

## Technologies Used
- **Backend**: Python, Flask
- **Data Mining / ML**: Pandas, NumPy, Scikit-learn (KNN)
- **Frontend**: HTML5, CSS3, Bootstrap 5, Chart.js

## Project Structure
- `app.py`: The main Flask web application containing all routes.
- `data_processor.py`: Contains the `MovieRecommender` class responsible for loading, cleaning, aggregating data, and training the KNN model.
- `requirements.txt`: Lists all Python dependencies.
- `templates/`: HTML templates for the web pages (`base.html`, `index.html`, `analysis.html`, `recommendation.html`).
- `static/css/style.css`: Custom styling for the application.

## Dataset
The system utilizes the [MovieLens Dataset](https://grouplens.org/datasets/movielens/). 
- For performance and optimization, only the first 200,000 rows of `ratings.csv` are used.
- The `movies.csv` and `ratings.csv` files should be placed in the parent directory (`../`) relative to this project folder, or you can update the paths in `app.py`.

## Installation and Running Instructions

1. **Prerequisites**: Ensure you have Python 3.8+ installed.
2. **Navigate to the Project Directory**:
   ```bash
   cd movie_recommender
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Application**:
   ```bash
   python app.py
   ```
5. **Access the Web App**:
   Open your web browser and navigate to `http://127.0.0.1:5000/`.

## Data Preprocessing Steps Performed
1. **Loading Data**: Read using Pandas.
2. **Cleaning**: Removed missing values and duplicate records.
3. **Feature Engineering**: Extracted the movie year from the title.
4. **Aggregation**: Calculated average rating and total rating count for each movie.
5. **Model Preparation**: Created a pivot table for users and movies and trained a KNN model using cosine similarity.
