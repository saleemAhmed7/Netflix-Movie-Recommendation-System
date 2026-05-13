from flask import Flask, jsonify, render_template, request
from data_processor import MovieRecommender
import os
import socket
import sys

import threading
import time
app = Flask(__name__)
debug_mode = os.environ.get("FLASK_DEBUG", "").lower() in {"1", "true", "yes"}
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0 if debug_mode else None
HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
PORT = int(os.environ.get("FLASK_PORT", "5000"))

# Initialize recommender with dataset paths
RATINGS_PATH = os.path.join(os.path.dirname(__file__), '..', 'ratings.csv')
MOVIES_PATH = os.path.join(os.path.dirname(__file__), '..', 'movies.csv')
LINKS_PATH = os.path.join(os.path.dirname(__file__), '..', 'links.csv')
PROJECT_DIR = os.path.dirname(__file__)

recommender = MovieRecommender(RATINGS_PATH, MOVIES_PATH, LINKS_PATH)
recommender_ready = False

def initialize_recommender():
    """Load movie data once before serving requests."""
    global recommender_ready
    if recommender_ready:
        return

    print("Starting application and initializing data...")
    try:
        recommender.preprocess()
        recommender_ready = True
        print("Application ready!")
    except Exception as exc:
        recommender_ready = False
        print(f"Application failed to initialize: {exc}", file=sys.stderr)
        raise


if not debug_mode or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    initialize_recommender()
    # Start dataset watcher only in the reloader child (or when not debugging)
    try:
        _start_dataset_watcher(_paths_to_watch)
    except Exception:
        pass

@app.before_request
def ensure_recommender_ready():
    """Keep routes usable even if the dev reloader imports the parent process first."""
    if request.path.startswith("/_dev/"):
        return
    initialize_recommender()

@app.context_processor
def inject_dev_flags():
    return {"dev_auto_reload": debug_mode}


# Simple translations for English, Arabic, and Turkish.
TRANSLATIONS = {
    "en": {
        "Home": "Home",
        "Dataset Analysis": "Dataset Analysis",
        "Get Recommendations": "Get Recommendations",
        "Data Mining Project": "Data Mining Project",
        "Discover Your Next": "Discover Your Next",
        "Cinematic Experience": "Cinematic Experience",
        "View Dataset Stats": "View Dataset Stats",
        "System Architecture": "System Architecture",
        "Find Your Next Watch": "Find Your Next Watch",
        "Filters": "Filters",
        "Based on a movie you like": "Based on a movie you like",
        "Genre": "Genre",
        "Minimum Rating (0-5)": "Minimum Rating (0-5)",
        "From Year": "From Year",
        "To Year": "To Year",
        "Results Count": "Results Count",
        "Generate Recommendations": "Generate Recommendations",
        "Recommended For You": "Recommended For You",
        "Awaiting Your Search": "Awaiting Your Search",
        "Powered by K-Nearest Neighbors (KNN) Recommendation Engine": "Powered by K-Nearest Neighbors (KNN) Recommendation Engine",
        "Top Rated Movies": "Top Rated Movies",
        "Trending Now (Most Popular)": "Trending Now (Most Popular)",
    },
    "ar": {
        "Home": "الرئيسية",
        "Dataset Analysis": "تحليل البيانات",
        "Get Recommendations": "الحصول على اقتراحات",
        "Data Mining Project": "مشروع تنقيب البيانات",
        "Discover Your Next": "اكتشف مشاهدتك التالية",
        "Cinematic Experience": "تجربة سينمائية",
        "View Dataset Stats": "عرض إحصائيات البيانات",
        "System Architecture": "بُنية النظام",
        "Find Your Next Watch": "ابحث عن مشاهدتك القادمة",
        "Filters": "مرشحات",
        "Based on a movie you like": "استنادًا إلى فيلم تحبه",
        "Genre": "النوع",
        "Minimum Rating (0-5)": "الحد الأدنى للتقييم (0-5)",
        "From Year": "من سنة",
        "To Year": "إلى سنة",
        "Results Count": "عدد النتائج",
        "Generate Recommendations": "إنشاء اقتراحات",
        "Recommended For You": "مقترح لك",
        "Awaiting Your Search": "بانتظار بحثك",
        "Powered by K-Nearest Neighbors (KNN) Recommendation Engine": "مدعوم بخوارزمية K-الأقرب (KNN)",
        "Top Rated Movies": "الأفلام الأعلى تقييمًا",
        "Trending Now (Most Popular)": "الرائج الآن (الأكثر شعبية)",
    },
    "tr": {
        "Home": "Anasayfa",
        "Dataset Analysis": "Veri Analizi",
        "Get Recommendations": "Öneri Al",
        "Data Mining Project": "Veri Madenciliği Projesi",
        "Discover Your Next": "Bir Sonrakini Keşfet",
        "Cinematic Experience": "Sinematik Deneyim",
        "View Dataset Stats": "Veri İstatistiklerini Gör",
        "System Architecture": "Sistem Mimarisi",
        "Find Your Next Watch": "Bir Sonraki İzlemeni Bul",
        "Filters": "Filtreler",
        "Based on a movie you like": "Beğendiğin bir filme göre",
        "Genre": "Tür",
        "Minimum Rating (0-5)": "Minimum Puan (0-5)",
        "From Year": "Başlangıç Yılı",
        "To Year": "Bitiş Yılı",
        "Results Count": "Sonuç Sayısı",
        "Generate Recommendations": "Önerileri Oluştur",
        "Recommended For You": "Senin İçin Öneriler",
        "Awaiting Your Search": "Aramanızı Bekliyor",
        "Powered by K-Nearest Neighbors (KNN) Recommendation Engine": "K-En Yakın Komşular (KNN) ile desteklenir",
        "Top Rated Movies": "En Yüksek Puanlı Filmler",
        "Trending Now (Most Popular)": "Şu Anda Trend Olanlar (En Popüler)",
    },
}


def _get_lang_from_request():
    # First check cookie, then Accept-Language header, default to English
    lang = request.cookies.get("lang") if request else None
    if lang and lang in TRANSLATIONS:
        return lang
    # check Accept-Language
    header = request.headers.get("Accept-Language", "") if request else ""
    if header:
        for part in header.split(','):
            code = part.split(';')[0].strip().lower()
            if code.startswith('ar'):
                return 'ar'
            if code.startswith('tr'):
                return 'tr'
            if code.startswith('en'):
                return 'en'
    return 'en'


@app.context_processor
def inject_translations():
    def t(key):
        try:
            lang = _get_lang_from_request()
            return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)
        except Exception:
            return key

    def static_url(filename):
        from flask import url_for

        if debug_mode:
            # append timestamp to bust cache during development
            ts = int(time.time())
            return f"{url_for('static', filename=filename)}?v={ts}"
        return url_for('static', filename=filename)

    return {
        "t": t,
        "current_lang": _get_lang_from_request(),
        "available_langs": ['en', 'ar', 'tr'],
        "static_url": static_url,
    }


@app.after_request
def add_dev_headers(response):
    # Disable caching in debug mode so CSS/JS/HTML changes appear immediately
    if debug_mode:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response
def latest_project_mtime():
    watched_paths = [
        os.path.join(PROJECT_DIR, "app.py"),
        os.path.join(PROJECT_DIR, "data_processor.py"),
        os.path.join(PROJECT_DIR, "templates"),
        os.path.join(PROJECT_DIR, "static"),
    ]
    latest_mtime = 0.0
    for path in watched_paths:
        if os.path.isfile(path):
            latest_mtime = max(latest_mtime, os.path.getmtime(path))
            continue
        for root, _, files in os.walk(path):
            for filename in files:
                if filename.endswith((".html", ".css", ".js", ".py")):
                    latest_mtime = max(latest_mtime, os.path.getmtime(os.path.join(root, filename)))
    return latest_mtime


# Start a background watcher thread to reload dataset files when they change.
# This runs regardless of debug mode so dataset updates reflect live.
def _start_dataset_watcher(paths, interval=5):
    data_lock = threading.Lock()

    def _watch_dataset_files():
        last_mtimes = {}
        for p in paths:
            try:
                last_mtimes[p] = os.path.getmtime(p) if p and os.path.exists(p) else None
            except Exception:
                last_mtimes[p] = None

        while True:
            time.sleep(interval)
            changed = False
            for p in paths:
                try:
                    m = os.path.getmtime(p) if p and os.path.exists(p) else None
                except Exception:
                    m = None
                if last_mtimes.get(p) != m:
                    print(f"Detected change in dataset file: {p}")
                    last_mtimes[p] = m
                    changed = True

            if changed:
                try:
                    with data_lock:
                        print("Reloading dataset due to file changes...")
                        recommender.preprocess()
                        print("Reload complete.")
                except Exception as e:
                    print(f"Failed to reload dataset: {e}")

    _watcher = threading.Thread(target=_watch_dataset_files, daemon=True)
    _watcher.start()


# Begin watching core dataset CSVs
_paths_to_watch = [RATINGS_PATH, MOVIES_PATH, LINKS_PATH]
_start_dataset_watcher(_paths_to_watch)

@app.route('/_dev/reload-token')
def dev_reload_token():
    if not debug_mode:
        return jsonify({"token": None})
    return jsonify({"token": latest_project_mtime()})

@app.route('/')
def home():
    """Renders the Home Page"""
    stats = recommender.get_stats()
    top_rated = stats.get('top_rated', []) if stats else []
    return render_template('index.html', top_rated=top_rated)

@app.route('/analysis')
def analysis():
    """Renders the Dataset Analysis Page"""
    stats = recommender.get_stats()
    return render_template('analysis.html', stats=stats)

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    """Renders the Recommendation Page"""
    all_genres = recommender.get_all_genres()
    all_years = recommender.get_all_years()
    
    recommendations = []
    
    if request.method == 'POST':
        movie_name = request.form.get('movie_name', '')
        genre = request.form.get('genre', '')
        min_rating = request.form.get('min_rating', '0')
        year_start = request.form.get('year_start', '')
        year_end = request.form.get('year_end', '')
        n_recommendations = int(request.form.get('n_recommendations', 5))
        
        recommendations = recommender.recommend(
            movie_name=movie_name if movie_name else None,
            genre=genre if genre else None,
            min_rating=float(min_rating) if min_rating else 0.0,
            year_start=year_start if year_start else None,
            year_end=year_end if year_end else None,
            n_recommendations=n_recommendations
        )
        
    return render_template(
        'recommendation.html', 
        genres=all_genres, 
        years=all_years,
        recommendations=recommendations
    )

def get_local_ipv4():
    """Return the LAN/hotspot IPv4 address that another device can open."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return "127.0.0.1"

if __name__ == '__main__':
    local_ipv4 = get_local_ipv4()
    print("\nOpen on this computer:")
    print(f"  http://127.0.0.1:{PORT}")
    print("Open on your phone while connected to the same hotspot/Wi-Fi:")
    print(f"  http://{local_ipv4}:{PORT}\n")
    app.run(debug=debug_mode, host=HOST, port=PORT, use_reloader=debug_mode, threaded=True)
