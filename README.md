# Movie Recommendation System

A beginner-friendly **content-based movie recommendation engine** built in Python. The system recommends movies by comparing a user's favorite genre against movie genre profiles using **cosine similarity**.

This project is designed as a portfolio-ready starter project suitable for internship evaluation, demonstrating core recommendation concepts without heavy machine learning dependencies.

---

## Features

- Curated movie dataset with multiple genres per title
- Interactive command-line interface (CLI)
- Genre-based similarity scoring using cosine similarity
- Top-N ranked recommendations with readable output
- Input validation for empty or invalid genre entries
- Fully commented Python source code

---

## Project Structure

```
movie-recommendation-system/
├── recommendation.py   # Main application and recommendation logic
├── movies.csv          # Movie dataset
├── requirements.txt    # Dependency notes
└── README.md           # Project documentation
```

---

## Dataset Structure

The dataset is stored in `movies.csv` with the following columns:

| Column  | Type   | Description                                      |
|---------|--------|--------------------------------------------------|
| `title` | string | Movie title                                      |
| `year`  | int    | Release year                                     |
| `rating`| float  | IMDb-style rating (used as a tie-breaker boost)   |
| `genres`| string | Pipe-separated genres (e.g., `Action|Sci-Fi`) |

### Supported Genres

- Action
- Comedy
- Horror
- Sci-Fi
- Romance
- Thriller

Movies may belong to one or more genres. Multi-genre titles improve recommendation diversity because similarity is computed across the full genre vector.

---

## Recommendation Logic

This project uses a **content-based filtering** approach.

### Step 1: Genre Vectorization

Each movie is converted into a binary feature vector based on supported genres.

Example for `Inception` (`Action|Sci-Fi|Thriller`):

```
[Action, Comedy, Horror, Sci-Fi, Romance, Thriller]
[  1  ,   0   ,   0   ,   1   ,   0    ,    1    ]
```

### Step 2: User Preference Vector

The user's favorite genre is encoded as a one-hot vector.

Example: favorite genre = `Sci-Fi`

```
[0, 0, 0, 1, 0, 0]
```

### Step 3: Cosine Similarity

Cosine similarity measures the angle between two vectors:

```
similarity = (A · B) / (||A|| × ||B||)
```

- **1.0** → perfect genre alignment
- **0.5** → partial overlap (multi-genre matches)
- **0.0** → no shared genres

### Step 4: Ranking Boost

A small rating adjustment is added to break ties fairly:

```
final_score = cosine_similarity + (rating / 100)
```

### Step 5: Top-N Output

Movies are sorted by `final_score` (and rating as secondary key), then the top 5 recommendations are displayed.

---

## How to Run

### Prerequisites

- Python 3.8 or newer
- No external libraries required

### Run the CLI

```bash
cd movie-recommendation-system
python3 recommendation.py
```

### Example Session

```
============================================================
        MOVIE RECOMMENDATION SYSTEM
   Content-Based Genre Similarity Engine
============================================================

Loaded 35 movies from dataset.

Available genres:
  1. Action
  2. Comedy
  3. Horror
  4. Sci-Fi
  5. Romance
  6. Thriller

Enter your favorite genre (name or number): Sci-Fi

Top recommendations for fans of Sci-Fi:
...
```

You can enter a genre by name (`Horror`) or by number (`3`).

---

## Invalid Input Handling

The CLI handles common input issues:

- Empty input → prompts again
- Unknown genre names → error message and retry
- Out-of-range numeric choices → error message and retry
- Missing/invalid dataset file → graceful exit with error message

---

## Future Improvements

1. **Collaborative Filtering** – recommend movies based on similar users' ratings
2. **TF-IDF Weighting** – weight rare genres more strongly than common ones
3. **User Profiles** – allow multiple favorite genres with custom weights
4. **Web Interface** – build a Flask/FastAPI frontend for easier demos
5. **Larger Dataset** – integrate TMDB/IMDb APIs for thousands of titles
6. **Matrix Factorization** – use SVD or neural embeddings for advanced ranking
7. **Explainability Panel** – show why each movie was recommended
8. **Unit Tests** – add automated tests for similarity and input validation

---

## Author Notes

This project intentionally keeps the implementation readable and educational. It demonstrates practical AI/recommendation fundamentals—feature encoding, vector similarity, ranking, and user-facing validation—making it a strong foundation for internship discussions and further expansion.
