# Project Report: Netflix-Style Movie Recommendation System

## 1. Project Overview

This project is a web-based movie recommendation system built with Flask and the MovieLens dataset. The application combines data preprocessing, K-Nearest Neighbors recommendation logic, dataset analysis, IMDb integration, and a Netflix-style user interface.

The goal of the system is to help users discover movies through two main experiences:

- Dataset analysis with ranked movie rows such as Top Rated Movies and Trending Now.
- Personalized recommendations based on a liked movie, genre, rating, year range, and number of results.

The project was improved without rebuilding the original application from scratch. Existing Flask routes, recommendation logic, IMDb support, and the dark cinematic UI were preserved.

## 2. Main Project Files

- `app.py`: Defines the Flask application, routes, startup flow, and development reload endpoint.
- `data_processor.py`: Handles dataset loading, preprocessing, statistics, KNN matrix creation, recommendation logic, IMDb links, poster fallback images, trailer links, and preprocessing cache.
- `templates/base.html`: Shared page layout, navigation, footer, scripts, and development auto-refresh logic.
- `templates/index.html`: Home page.
- `templates/analysis.html`: Dataset analysis page with Netflix-style ranked sliders.
- `templates/recommendation.html`: Recommendation form and recommendation result cards.
- `static/css/style.css`: Main visual styling for the Netflix-style interface.
- `requirements.txt`: Python dependencies required to run the project.

## 3. Dataset Description

The project uses MovieLens-style CSV files:

- `movies.csv`: Contains `movieId`, `title`, and `genres`.
- `ratings.csv`: Contains user ratings and is the main source for recommendation modeling.
- `links.csv`: Contains `movieId`, `imdbId`, and `tmdbId`.
- `tags.csv`: Exists in the dataset but is not required for the current recommendation workflow.

The `links.csv` file is used to generate IMDb URLs for each movie. The movie title and year are used to generate YouTube trailer search links.

## 4. Data Preprocessing

The preprocessing pipeline in `data_processor.py` performs these steps:

1. Loads only required columns from the large ratings file to reduce memory usage.
2. Normalizes column names such as `userId`, `movieId`, and `rating`.
3. Loads movie metadata from `movies.csv`.
4. Loads IMDb/TMDb IDs from `links.csv`.
5. Merges movie data with IMDb link data.
6. Extracts release year from titles such as `Toy Story (1995)`.
7. Cleans movie titles by removing the year suffix.
8. Converts genre formatting from pipe-separated text to readable comma-separated text.
9. Calculates average rating and total rating count for every movie.
10. Assigns poster fallback images based on the movie genre.
11. Builds lookup dictionaries for fast title and movie ID access.

## 5. Recommendation Algorithm

The system uses K-Nearest Neighbors with cosine distance. Each movie is represented as a sparse vector of user ratings. When a user enters a movie name, the system finds the closest movie vectors and returns similar movies.

The recommendation process supports:

- Similarity by movie title.
- Genre filtering.
- Minimum rating filtering.
- Release year range filtering.
- Configurable number of recommendations.

If a movie name is not provided or no KNN recommendations pass the filters, the system falls back to top-rated movies that match the selected filters.

## 6. KNN and Memory Optimization

The original dataset is large, so the recommendation system uses several optimizations:

- Uses Pandas typed columns such as `int32` and `float32`.
- Uses SciPy sparse matrices instead of dense pivot tables.
- Limits the KNN model to popular movies with enough ratings.
- Avoids repeated expensive lookups by using dictionaries and indexed DataFrames.
- Caches preprocessed artifacts on disk.

The cache stores:

- Clean movie metadata.
- Dataset summary.
- Sparse KNN matrix.
- KNN movie IDs.
- Source file signature to detect stale cache.

This makes later application startups much faster.

## 7. Flask Startup and Runtime Stability

The Flask startup path was improved to avoid instability and `ERR_CONNECTION_REFUSED`.

The issue came from the combination of:

- A very large `ratings.csv` file.
- Expensive preprocessing at application startup.
- Flask debug reloader starting multiple Python processes.

Fixes added:

- Preprocessing cache to avoid rebuilding data every launch.
- Safer initialization guard around the Flask debug reloader.
- Flask binds to `http://127.0.0.1:5000`.
- `start.bat` enables development mode with `FLASK_DEBUG=1`.
- A development reload endpoint was added for local browser refresh.

## 8. Development Auto Reload

The project now supports a smoother local development workflow:

- Python changes restart Flask through the debug reloader.
- Template and CSS changes trigger automatic browser refresh through `/_dev/reload-token`.
- Static file caching is disabled during debug mode.

This helps changes appear quickly while developing the UI.

## 9. Netflix-Style UI Improvements

The analysis page was enhanced to look and behave more like a Netflix movie row.

Implemented UI improvements:

- Fixed broken Top Rated Movies card rendering.
- Added visible movie cards inside horizontal sliders.
- Added movie poster, title, rating, genre, release year, and vote count.
- Added large ranking badges for the first 10 movies.
- Reduced ranking badge size after visual review.
- Added left and right arrow controls for slider navigation.
- Preserved hover animations and dark cinematic styling.

The ranking badges appear on:

- Top Rated Movies.
- Trending Now.

The styling is reusable for future ranked movie rows.

## 10. IMDb and Trailer Integration

IMDb integration:

- Each movie uses its `imdbId` from `links.csv`.
- The system generates links in this format:

```text
https://www.imdb.com/title/tt0111161/
```

Trailer integration:

- The dataset does not include direct official trailer video IDs.
- The Trailer button opens YouTube search directly using the movie title, year, and `official trailer`.
- This avoids needing a YouTube API key and gives reliable search results for each movie.

Example:

```text
https://www.youtube.com/results?search_query=Toy+Story+1995+official+trailer
```

## 11. Recommendation Page Improvements

The recommendation cards were updated to match the analysis cards:

- Movie poster.
- Title.
- Rating.
- Release year.
- Genres.
- Recommendation reason.
- IMDb button.
- Trailer button.

The KNN recommendation logic was preserved.

## 12. Requirements

The project dependencies are listed in `requirements.txt`:

- Flask
- pandas
- numpy
- scipy
- scikit-learn

`scipy` is required because the project uses sparse matrices through `scipy.sparse`.

## 13. Preserved Features

The following features were preserved:

- Flask routes for home, analysis, and recommendation pages.
- KNN recommendation system.
- MovieLens dataset loading.
- IMDb link generation.
- Netflix-style dark UI.
- Recommendation filters.
- Horizontal sliders and movie cards.
- Development startup through `start.bat`.

## 14. Final Result

The final project is a complete Flask-based movie recommendation system with:

- Efficient dataset preprocessing.
- Cached KNN recommendation artifacts.
- Stable local startup.
- Netflix-style ranked movie sliders.
- IMDb and trailer buttons.
- Functional recommendation filters.
- A written project report stored directly in the project files.

## 15. Recent Development & Developer Experience Improvements

These updates were added to make development faster and to improve robustness when adapting the project to a new dataset. Changes are non-destructive and preserve all existing features (recommendation system, KNN logic, Netflix UI, charts, sliders, IMDb/trailer buttons).

- Automatic dataset watcher: `app.py` starts a background watcher thread that polls the key CSV files (`ratings.csv`, `movies.csv`, `links.csv`) and automatically calls the preprocessing pipeline when files change — no need to stop/start the server to pick up dataset updates.
- Safe reloader integration: the watcher and preprocessing are started only in the correct Flask process (avoids duplicate startup while using the debug reloader).
- Development hot-reload helpers:
	- A `/ _dev/reload-token` endpoint and a small client script in `base.html` that auto-refreshes the browser when templates or static files change.
	- `after_request` response headers disable caching in debug mode so CSS/JS changes appear immediately.
	- A `static_url()` helper exposed to templates that appends a timestamp to static asset URLs during development to bust browser caches.
- Robust data handling in `data_processor.py`:
	- Flexible IMDb link parsing (handles `tt` ids, numeric ids, and full IMDb URLs).
	- Poster URL fallbacks per-genre and support for `poster_url` column if present.
	- Safer numeric coercion for IDs and guard clauses that avoid KNN training when insufficient data is present.
	- Preprocessing cache: saves/loads pickled movie metadata, sparse KNN matrix, and identifiers to speed up repeated starts.
- i18n / Language selector:
	- Simple translation helper `t()` added in `app.py` with English (`en`), Arabic (`ar`), and Turkish (`tr`) mappings.
	- Language selector dropdown in the navbar that sets a `lang` cookie; templates use `t('key')` for translatable strings.
	- Hover-to-open language menu for desktop and click-to-open for mobile — responsive and preserves the Netflix-style design.
- UI and accessibility fixes:
	- Fixed Bootstrap JS include so dropdowns and other interactive components work correctly.
	- CSS fixes to ensure dropdown menus render above the navbar (`z-index`, `overflow: visible`) and include smooth open/close animation.
	- Responsive dropdown behavior for mobile (full-width, easy to tap).

## 16. How to run in development (recommended)

1. Open a terminal in the project root (where `app.py` is located).
2. Set debug mode and run the app:

Windows CMD:
```cmd
set FLASK_DEBUG=1
python app.py
```

PowerShell:
```powershell
$env:FLASK_DEBUG=1; python app.py
```

3. Open `http://127.0.0.1:5000` in your browser. While developing:
	- Save Python files to trigger the Flask auto-reloader.
	- Save template (`.html`) and static (`.css`, `.js`) files; the page will auto-refresh or load updated assets because caching is disabled in debug mode and `static_url()` appends a timestamp to static assets.
	- Update dataset CSV files and the background watcher will re-run preprocessing automatically; changes appear on the site without restarting the server.

If you see layout caching or stale assets, do a hard refresh (Ctrl+F5) or open the page in an incognito window.

This report documents the system, the improvements made, and the final structure of the project without adding an extra report page to the website UI.
