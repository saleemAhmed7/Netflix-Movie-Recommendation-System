import os
import re
import json
import time
from functools import lru_cache
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz, save_npz
from sklearn.neighbors import NearestNeighbors


class MovieRecommender:
    CACHE_VERSION = 7

    def __init__(self, ratings_path, movies_path, links_path=None, tags_path=None):
        self.ratings_path = ratings_path
        self.movies_path = movies_path
        self.links_path = links_path
        self.tags_path = tags_path

        self.model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.movie_features = None
        self.knn_movie_ids = None
        self.movies_clean = None
        self.merged_df = None
        self.dataset_summary = {}
        self.movieid_to_title = {}
        self.title_to_movieid = {}
        self.movies_by_id = None
        self.knn_movie_id_set = set()

        self.min_votes_for_popular = 50
        self.min_votes_for_toprated = 100
        self.max_knn_movies = 10000
        self.cache_dir = os.path.join(os.path.dirname(__file__), "cache")

    def preprocess(self):
        started_at = time.perf_counter()
        if self._load_cache():
            print(f"Loaded preprocessed movie data from cache in {time.perf_counter() - started_at:.1f}s.")
            return

        print("Loading new dataset files...")
        ratings = self._load_ratings()
        movies = self._load_movies()
        links = self._load_links()
        tags = self._load_tags()

        # Clean and prepare dataset
        movies, ratings = self._clean_and_prepare_data(movies, ratings, links)

        print("Recalculating movie statistics and rating analytics...")
        rating_stats = (
            ratings.groupby("movieId", observed=True)["rating"]
            .agg(avg_rating="mean", total_ratings="count")
            .reset_index()
        )
        rating_stats["avg_rating"] = rating_stats["avg_rating"].round(2)

        movies_clean = movies.merge(rating_stats, on="movieId", how="inner")
        movies_clean["avg_rating"] = movies_clean["avg_rating"].round(1)
        movies_clean["total_ratings"] = movies_clean["total_ratings"].astype(np.int32)
        movies_clean["poster_url"] = movies_clean.apply(self._poster_for_movie, axis=1)

        self.movies_clean = movies_clean
        self._build_lookup_tables()

        self.dataset_summary = {
            "total_users": int(ratings["userId"].nunique()),
            "total_movies": int(movies_clean["movieId"].nunique()),
            "total_ratings": int(len(ratings)),
            "total_tags": int(len(tags)) if tags is not None else 0,
            "rating_mean": float(ratings["rating"].mean()),
            "rating_min": float(ratings["rating"].min()),
            "rating_max": float(ratings["rating"].max()),
        }

        # Keep a compact merged view for legacy analysis code without duplicating every rating row.
        self.merged_df = rating_stats

        print("Rebuilding recommendation data and sparse KNN input matrix...")
        self._build_knn_matrix(ratings, rating_stats)
        self._save_cache()
        print(f"Preprocessing complete in {time.perf_counter() - started_at:.1f}s!")

    def _load_ratings(self):
        ratings = pd.read_csv(
            self.ratings_path,
            usecols=lambda col: col in {"userId", "user_id", "userID", "movieId", "movie_id", "movieID", "rating"},
            dtype={"userId": "int32", "movieId": "int32", "rating": "float32"},
        )
        ratings = ratings.rename(
            columns={
                "movie_id": "movieId",
                "movieID": "movieId",
                "user_id": "userId",
                "userID": "userId",
            }
        )
        self._require_columns(ratings, {"userId", "movieId", "rating"}, "ratings")
        ratings = ratings.dropna(subset=["userId", "movieId", "rating"])
        ratings["userId"] = ratings["userId"].astype(np.int32)
        ratings["movieId"] = ratings["movieId"].astype(np.int32)
        ratings["rating"] = ratings["rating"].astype(np.float32)
        return ratings

    def _load_movies(self):
        movies = pd.read_csv(self.movies_path)
        movies = movies.rename(
            columns={
                "movie_id": "movieId",
                "movieID": "movieId",
                "name": "title",
                "Title": "title",
                "genre": "genres",
            }
        )
        self._require_columns(movies, {"movieId", "title"}, "movies")
        if "genres" not in movies.columns:
            movies["genres"] = ""
        movies = movies.dropna(subset=["movieId", "title"]).drop_duplicates(subset=["movieId"])
        movies["movieId"] = movies["movieId"].astype(np.int32)
        return movies

    def _load_links(self):
        if not self.links_path or not os.path.exists(self.links_path):
            return None
        links = pd.read_csv(self.links_path)
        links = links.rename(
            columns={
                "movie_id": "movieId",
                "movieID": "movieId",
                "imdb_id": "imdbId",
                "tmdb_id": "tmdbId",
            }
        )
        if "movieId" not in links.columns:
            return None
        keep = [col for col in ["movieId", "imdbId", "tmdbId"] if col in links.columns]
        links = links[keep].drop_duplicates(subset=["movieId"])
        links["movieId"] = links["movieId"].astype(np.int32)
        return links

    def _load_tags(self):
        if not self.tags_path or not os.path.exists(self.tags_path):
            return None
        try:
            return pd.read_csv(self.tags_path, usecols=lambda col: col in {"userId", "movieId", "tag", "timestamp"})
        except Exception:
            return None

    def _clean_and_prepare_data(self, movies, ratings, links):
        print("Starting comprehensive data cleaning and validation...")
        initial_movies_count = len(movies)
        initial_ratings_count = len(ratings)
        
        # --- 1. Basic Validation ---
        movies = movies.dropna(subset=['movieId', 'title'])
        ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])
        
        # Strict IMDb ID Requirement
        if links is not None:
            links = links.dropna(subset=['movieId', 'imdbId'])
            links = links[links['imdbId'].astype(str).str.contains(r'\d+', na=False)]
            movies = movies.merge(links, on="movieId", how="inner")
        
        # --- 2. Movie Quality Improvements ---
        movies["raw_title"] = movies["title"].astype(str)
        movies["year"] = movies["raw_title"].str.extract(r"\((\d{4})\)\s*$")[0]
        movies["title"] = movies["raw_title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()
        
        # Remove movies missing year
        movies = movies.dropna(subset=['year'])
        
        # Clean genres and remove missing/invalid
        movies["genres"] = (
            movies["genres"]
            .fillna("")
            .astype(str)
            .str.replace("|", ", ", regex=False)
            .str.replace("(no genres listed)", "", regex=False)
            .str.strip()
        )
        movies = movies[movies["genres"] != ""]
        
        # Detect and remove duplicates
        movies = movies.drop_duplicates(subset=['movieId'], keep='first')
        movies["norm_title"] = movies["title"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
        
        movies_before_dedup = len(movies)
        movies = movies.drop_duplicates(subset=['norm_title', 'year'], keep='first')
        movies_duplicates_removed = movies_before_dedup - len(movies)
        
        if "imdbId" in movies.columns:
            movies["imdb_link"] = movies["imdbId"].apply(self._imdb_url)
        else:
            movies["imdb_link"] = "#"
            
        movies = movies.drop(columns=["raw_title", "norm_title"])
        
        # --- 3. Ratings Quality Improvements ---
        ratings['userId'] = pd.to_numeric(ratings['userId'], errors='coerce')
        ratings['movieId'] = pd.to_numeric(ratings['movieId'], errors='coerce')
        ratings['rating'] = pd.to_numeric(ratings['rating'], errors='coerce')
        ratings = ratings.dropna(subset=['userId', 'movieId', 'rating'])
        
        # Remove duplicate ratings
        ratings_before_dedup = len(ratings)
        ratings = ratings.drop_duplicates(subset=['userId', 'movieId'], keep='last')
        ratings_duplicates_removed = ratings_before_dedup - len(ratings)
        
        # Remove ratings for orphaned movies
        valid_movie_ids = set(movies['movieId'])
        ratings_before_integrity = len(ratings)
        ratings = ratings[ratings['movieId'].isin(valid_movie_ids)]
        ratings_invalid_removed = ratings_before_integrity - len(ratings)
        
        final_movies_count = len(movies)
        final_ratings_count = len(ratings)
        
        print("\n=== Data Cleaning & Quality Report ===")
        print("Movies Validation:")
        print(f"- Total records before cleaning: {initial_movies_count}")
        print(f"- Duplicate records removed: {movies_duplicates_removed}")
        print(f"- Invalid records removed: {initial_movies_count - final_movies_count - movies_duplicates_removed}")
        print(f"- Remaining valid records: {final_movies_count}")
        
        print("\nRatings Validation:")
        print(f"- Total records before cleaning: {initial_ratings_count}")
        print(f"- Duplicate records removed: {ratings_duplicates_removed}")
        print(f"- Invalid records removed: {ratings_invalid_removed}")
        print(f"- Remaining valid records: {final_ratings_count}")
        print("======================================\n")
        
        return movies, ratings

    def _build_knn_matrix(self, ratings, rating_stats):
        eligible = rating_stats[rating_stats["total_ratings"] >= self.min_votes_for_popular]
        eligible = eligible.sort_values(["total_ratings", "avg_rating"], ascending=False).head(self.max_knn_movies)
        eligible_movie_ids = eligible["movieId"].astype(np.int32).to_numpy()

        knn_ratings = ratings[ratings["movieId"].isin(eligible_movie_ids)]
        if knn_ratings.empty:
            self.movie_features = csr_matrix((0, 0), dtype=np.float32)
            self.knn_movie_ids = np.array([], dtype=np.int32)
            return

        movie_codes, movie_ids = pd.factorize(knn_ratings["movieId"], sort=True)
        user_codes, _ = pd.factorize(knn_ratings["userId"], sort=True)
        self.knn_movie_ids = movie_ids.astype(np.int32)
        self.knn_movie_id_set = set(self.knn_movie_ids.tolist())
        user_count = int(user_codes.max()) + 1
        self.movie_features = csr_matrix(
            (knn_ratings["rating"].astype(np.float32), (movie_codes, user_codes)),
            shape=(len(movie_ids), user_count),
            dtype=np.float32,
        )
        self.model_knn.fit(self.movie_features)

    def get_stats(self):
        if self.movies_clean is None:
            return {}

        top_rated = (
            self.movies_clean[self.movies_clean["total_ratings"] >= self.min_votes_for_toprated]
            .sort_values(["avg_rating", "total_ratings"], ascending=False)
            .head(10)
        )
        most_rated = self.movies_clean.sort_values(["total_ratings", "avg_rating"], ascending=False).head(10)
        recent_movies = (
            self.movies_clean[self.movies_clean["year"].astype(str).str.isdigit()]
            .sort_values(["year", "total_ratings"], ascending=False)
            .head(20)
        )

        genre_counts = {}
        all_genres = []
        for genres_str in self.movies_clean["genres"].fillna(""):
            all_genres.extend(self._split_genres(genres_str))
        if all_genres:
            genre_counts = pd.Series(all_genres).value_counts().head(10).to_dict()

        return {
            **self.dataset_summary,
            "top_rated": self._records(top_rated),
            "most_rated": self._records(most_rated),
            "recent_movies": self._records(recent_movies),
            "genre_counts": genre_counts,
        }

    def get_all_genres(self):
        if self.movies_clean is None:
            return []
        genres_set = set()
        for genres_str in self.movies_clean["genres"].dropna():
            genres_set.update(self._split_genres(genres_str))
        return sorted(genres_set)

    def get_all_years(self):
        if self.movies_clean is None:
            return []
        years = self.movies_clean["year"][self.movies_clean["year"] != "Unknown"].astype(str).unique()
        return sorted([int(y) for y in years if y.isdigit()])

    def get_genre_movie_count(self, selected_genres=None):
        """Return the number of movies matching ALL selected genres (AND logic).
        If no genres are selected, return the total movie count."""
        if self.movies_clean is None:
            return 0
        if not selected_genres:
            return len(self.movies_clean)
        selected_lower = {str(g).strip().lower() for g in selected_genres if str(g).strip()}
        if not selected_lower:
            return len(self.movies_clean)

        def has_all_genres(genres_str):
            if not isinstance(genres_str, str):
                return False
            movie_genres = {g.strip().lower() for g in re.split(r"[,|]", genres_str) if g.strip()}
            return selected_lower.issubset(movie_genres)

        return int(self.movies_clean["genres"].apply(has_all_genres).sum())

    def recommend(self, movie_name=None, genre=None, min_rating=0.0, year_start=None, year_end=None, n_recommendations=5):
        recommendations = []
        knn_total_candidates = 0
        knn_filtered_candidates = 0
        knn_recommended_ids = set()

        # Normalize selected genres once
        selected_lower = set()
        if isinstance(genre, (list, tuple, set)) and len(genre) > 0:
            selected_lower = {str(g).lower().strip() for g in genre if str(g).strip()}

        # Pre-filter all movies according to criteria
        filtered_movies = self._filter_movies(self.movies_clean.copy(), genre, min_rating, year_start, year_end)
        valid_movie_ids = set(filtered_movies["movieId"].tolist())

        if movie_name and self.movie_features is not None and self.movie_features.shape[0] > 0:
            mid = self._find_movie_id(movie_name)
            if mid is not None and mid in self.knn_movie_id_set:
                movie_idx = int(np.where(self.knn_movie_ids == mid)[0][0])

                distances, indices = self.model_knn.kneighbors(
                    self.movie_features[movie_idx],
                    n_neighbors=self.movie_features.shape[0],
                )

                raw_recommendations = []
                removed_count = 0

                for i, idx in enumerate(indices.flatten()):
                    rec_mid = int(self.knn_movie_ids[idx])
                    if rec_mid == mid:
                        continue

                    knn_total_candidates += 1

                    # Pre-filter enforces strict validity
                    if rec_mid not in valid_movie_ids:
                        removed_count += 1
                        continue

                    movie_details = self.movies_by_id.loc[rec_mid]
                    rec = self._movie_record(movie_details)
                    rec["reason"] = "Recommended using KNN similarity matching"

                    # Post-KNN AND validation: movie must contain ALL selected genres
                    if selected_lower:
                        movie_genres = self._split_genres(movie_details.get("genres", ""))
                        movie_genres_lower = {g.lower().strip() for g in movie_genres}

                        if not selected_lower.issubset(movie_genres_lower):
                            removed_count += 1
                            continue

                        knn_filtered_candidates += 1
                        # Reorder so matched genres appear first in UI
                        matched = [g for g in movie_genres if g.lower().strip() in selected_lower]
                        unmatched = [g for g in movie_genres if g.lower().strip() not in selected_lower]
                        rec["genres"] = ", ".join(matched + unmatched)
                        rec["reason"] = "Matches selected genres"
                    else:
                        knn_filtered_candidates += 1

                    rec["_distance"] = float(distances.flatten()[i])
                    raw_recommendations.append(rec)
                    knn_recommended_ids.add(rec_mid)

                    if len(raw_recommendations) >= n_recommendations:
                        break

                recommendations = raw_recommendations[:n_recommendations]
                print(f"Total KNN candidates generated: {knn_total_candidates}")
                print(f"Candidates after genre filtering: {knn_filtered_candidates}")
                print(f"Final displayed results count: {len(recommendations)}")

        # Fill from remaining valid matches when KNN results are insufficient
        if len(recommendations) < n_recommendations:
            remaining = n_recommendations - len(recommendations)
            if selected_lower:
                fallback = filtered_movies.copy()
            else:
                fallback = filtered_movies[filtered_movies["total_ratings"] >= self.min_votes_for_popular].copy()

            fallback = fallback[~fallback["movieId"].isin(knn_recommended_ids)]

            if selected_lower:
                def reorder_genres(movie_genres_str):
                    if not isinstance(movie_genres_str, str):
                        return ""
                    movie_genres = self._split_genres(movie_genres_str)
                    matched = [g for g in movie_genres if g.lower().strip() in selected_lower]
                    unmatched = [g for g in movie_genres if g.lower().strip() not in selected_lower]
                    return ", ".join(matched + unmatched)

                fallback["genres"] = fallback["genres"].apply(reorder_genres)

            top_filtered = fallback.sort_values(["avg_rating", "total_ratings"], ascending=[False, False]).head(remaining)

            if selected_lower:
                reason = "Matches selected genres"
            elif genre or min_rating or year_start or year_end:
                reason = "Matches selected genre and rating filters"
            else:
                reason = "Similar users highly rated this movie"

            additional = self._records(top_filtered, reason=reason)
            recommendations.extend(additional)
            print(f"Final displayed results count: {len(recommendations)}")

        return recommendations

    def _filter_movies(self, movies, genre, min_rating, year_start, year_end):
        if genre:
            if not isinstance(genre, (list, tuple, set)):
                genre = [genre]
            
            # AND match: movie must contain ALL selected genres
            selected_lower = {str(g).strip().lower() for g in genre if str(g).strip()}
            
            if selected_lower:
                def has_all_genres(genres_str):
                    if not isinstance(genres_str, str):
                        return False
                    movie_genres = {g.strip().lower() for g in re.split(r"[,|]", genres_str) if g.strip()}
                    return selected_lower.issubset(movie_genres)

                movies = movies[movies["genres"].apply(has_all_genres)]
                
        if min_rating:
            movies = movies[movies["avg_rating"] >= float(min_rating)]
        if year_start:
            year_values = pd.to_numeric(movies["year"], errors="coerce")
            movies = movies[year_values >= int(year_start)]
        if year_end:
            year_values = pd.to_numeric(movies["year"], errors="coerce")
            movies = movies[year_values <= int(year_end)]
        return movies

    def _passes_filters(self, movie, min_rating, genre, year_start, year_end):
        return not self._filter_movies(pd.DataFrame([movie]), genre, min_rating, year_start, year_end).empty

    def _records(self, dataframe, reason=None):
        records = []
        for _, row in dataframe.iterrows():
            record = self._movie_record(row)
            if reason:
                record["reason"] = reason
            records.append(record)
        return records

    def _movie_record(self, row):
        return {
            "movieId": int(row["movieId"]),
            "title": row.get("title", ""),
            "genres": row.get("genres", ""),
            "year": row.get("year", "Unknown"),
            "avg_rating": round(float(row.get("avg_rating", 0)), 1),
            "total_ratings": int(row.get("total_ratings", 0)),
            "imdbId": row.get("imdbId", ""),
            "tmdbId": row.get("tmdbId", ""),
            "imdb_link": row.get("imdb_link", "#"),
            "trailer_url": self._youtube_trailer_search_url(row.get("title", ""), row.get("year", "Unknown")),
            "poster_url": row.get("poster_url", ""),
        }

    def get_movie_by_id(self, movie_id):
        if self.movies_by_id is None:
            return None
        try:
            return self._movie_record(self.movies_by_id.loc[int(movie_id)])
        except (KeyError, TypeError, ValueError):
            return None

    def _build_lookup_tables(self):
        self.movieid_to_title = self.movies_clean.set_index("movieId")["title"].to_dict()
        self.movies_by_id = self.movies_clean.set_index("movieId", drop=False)
        self.title_to_movieid = {
            self._normalize_title(row["title"]): int(row["movieId"])
            for _, row in self.movies_clean.iterrows()
            if row.get("title")
        }

    def _cache_paths(self):
        return {
            "manifest": os.path.join(self.cache_dir, "manifest.json"),
            "movies_clean": os.path.join(self.cache_dir, "movies_clean.pkl"),
            "dataset_summary": os.path.join(self.cache_dir, "dataset_summary.json"),
            "movie_features": os.path.join(self.cache_dir, "movie_features.npz"),
            "knn_movie_ids": os.path.join(self.cache_dir, "knn_movie_ids.npy"),
        }

    def _source_signature(self):
        signature = {"version": self.CACHE_VERSION, "files": {}}
        for label, path in {
            "ratings": self.ratings_path,
            "movies": self.movies_path,
            "links": self.links_path,
            "tags": self.tags_path,
        }.items():
            if path and os.path.exists(path):
                stat = os.stat(path)
                signature["files"][label] = {
                    "path": os.path.abspath(path),
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                }
            else:
                signature["files"][label] = None
        return signature

    def _load_cache(self):
        paths = self._cache_paths()
        required = paths.values()
        if not all(os.path.exists(path) for path in required):
            return False

        try:
            with open(paths["manifest"], "r", encoding="utf-8") as file:
                manifest = json.load(file)
            if manifest != self._source_signature():
                return False

            self.movies_clean = pd.read_pickle(paths["movies_clean"])
            with open(paths["dataset_summary"], "r", encoding="utf-8") as file:
                self.dataset_summary = json.load(file)
            self.movie_features = load_npz(paths["movie_features"])
            self.knn_movie_ids = np.load(paths["knn_movie_ids"])
            self.knn_movie_id_set = set(self.knn_movie_ids.tolist())
            self.model_knn.fit(self.movie_features)
            self._build_lookup_tables()
            self.merged_df = self.movies_clean[["movieId", "avg_rating", "total_ratings"]].copy()
            return True
        except Exception as exc:
            print(f"Ignoring stale or unreadable cache: {exc}")
            return False

    def _save_cache(self):
        paths = self._cache_paths()
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.movies_clean.to_pickle(paths["movies_clean"])
            with open(paths["dataset_summary"], "w", encoding="utf-8") as file:
                json.dump(self.dataset_summary, file)
            save_npz(paths["movie_features"], self.movie_features)
            np.save(paths["knn_movie_ids"], self.knn_movie_ids)
            with open(paths["manifest"], "w", encoding="utf-8") as file:
                json.dump(self._source_signature(), file)
        except Exception as exc:
            print(f"Could not write preprocessing cache: {exc}")

    def _find_movie_id(self, movie_name):
        normalized = self._normalize_title(movie_name)
        if normalized in self.title_to_movieid:
            return self.title_to_movieid[normalized]

        matches = self.movies_clean[
            self.movies_clean["title"].str.lower().str.contains(re.escape(normalized), na=False)
        ]
        if matches.empty:
            return None
        return int(matches.sort_values("total_ratings", ascending=False).iloc[0]["movieId"])

    @staticmethod
    def _require_columns(dataframe, columns, label):
        missing = columns - set(dataframe.columns)
        if missing:
            raise ValueError(f"{label} file is missing required columns: {', '.join(sorted(missing))}")

    @staticmethod
    def _normalize_title(title):
        return re.sub(r"\s+", " ", str(title).strip().lower())

    @staticmethod
    def _split_genres(genres_str):
        return [
            genre.strip()
            for genre in re.split(r"[,|]", str(genres_str))
            if genre.strip() and genre.strip().lower() != "(no genres listed)"
        ]

    @staticmethod
    def _imdb_url(imdb_id):
        if pd.isna(imdb_id) or str(imdb_id).strip() == "":
            return "#"
        digits = re.sub(r"\D", "", str(imdb_id))
        return f"https://www.imdb.com/title/tt{int(digits):07d}/" if digits else "#"

    @staticmethod
    def _youtube_trailer_search_url(title, year):
        query_parts = [str(title).strip()]
        if str(year).isdigit():
            query_parts.append(str(year))
        query_parts.append("official trailer")
        return f"https://www.youtube.com/results?search_query={quote_plus(' '.join(query_parts))}"

    @staticmethod
    @lru_cache(maxsize=128)
    def _genre_image(genre):
        images = {
            "Action": "https://images.unsplash.com/photo-1535016120720-40c646be5580?q=80&w=500&auto=format&fit=crop",
            "Adventure": "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?q=80&w=500&auto=format&fit=crop",
            "Animation": "https://images.unsplash.com/photo-1601933470928-c53e1792cd1b?q=80&w=500&auto=format&fit=crop",
            "Children": "https://images.unsplash.com/photo-1516627145497-ae6968895b74?q=80&w=500&auto=format&fit=crop",
            "Comedy": "https://images.unsplash.com/photo-1527224538127-2104bb71c51b?q=80&w=500&auto=format&fit=crop",
            "Crime": "https://images.unsplash.com/photo-1509248961158-e54f6934749c?q=80&w=500&auto=format&fit=crop",
            "Documentary": "https://images.unsplash.com/photo-1492619375914-88005aa9e8fb?q=80&w=500&auto=format&fit=crop",
            "Drama": "https://images.unsplash.com/photo-1485846234645-a62644f84728?q=80&w=500&auto=format&fit=crop",
            "Fantasy": "https://images.unsplash.com/photo-1518709268805-4e9042af2176?q=80&w=500&auto=format&fit=crop",
            "Horror": "https://images.unsplash.com/photo-1505635552518-3448ff116af3?q=80&w=500&auto=format&fit=crop",
            "Romance": "https://images.unsplash.com/photo-1518199266791-5375a83190b7?q=80&w=500&auto=format&fit=crop",
            "Sci-Fi": "https://images.unsplash.com/photo-1446776811953-b23d57bd21aa?q=80&w=500&auto=format&fit=crop",
            "Thriller": "https://images.unsplash.com/photo-1478720568477-152d9b164e26?q=80&w=500&auto=format&fit=crop",
        }
        return images.get(genre, "https://images.unsplash.com/photo-1536440136628-849c177e76a1?q=80&w=500&auto=format&fit=crop")

    def _poster_for_movie(self, row):
        if "poster_url" in row and pd.notna(row.get("poster_url")) and str(row.get("poster_url")).strip():
            return row.get("poster_url")
        return f"/poster/{int(row.get('movieId'))}.svg"
