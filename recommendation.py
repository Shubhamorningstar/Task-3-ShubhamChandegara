#!/usr/bin/env python3
"""
Movie Recommendation System
---------------------------
A beginner-friendly content-based recommendation engine that suggests
movies using cosine similarity over genre feature vectors.
"""

import csv
import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Supported genres used for vector encoding and user input validation
VALID_GENRES = [
    "Action",
    "Comedy",
    "Horror",
    "Sci-Fi",
    "Romance",
    "Thriller",
]

# Number of top recommendations to display
TOP_N = 5

# Path to the movie dataset (relative to this script)
DATASET_PATH = os.path.join(os.path.dirname(__file__), "movies.csv")


@dataclass
class Movie:
    """Represents a single movie record from the dataset."""

    title: str
    year: int
    rating: float
    genres: List[str]
    genre_vector: List[float]


def load_movies(filepath: str) -> List[Movie]:
    """
    Load movies from a CSV file and convert each row into a Movie object.

    Expected CSV columns: title, year, rating, genres
    Genres are pipe-separated (e.g., "Action|Sci-Fi").
    """
    movies: List[Movie] = []

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    with open(filepath, newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            # Parse pipe-separated genres and keep only supported genres
            raw_genres = row["genres"].split("|")
            genres = [genre.strip() for genre in raw_genres if genre.strip() in VALID_GENRES]

            if not genres:
                # Skip movies with no valid genres to avoid empty vectors
                continue

            movie = Movie(
                title=row["title"].strip(),
                year=int(row["year"]),
                rating=float(row["rating"]),
                genres=genres,
                genre_vector=build_genre_vector(genres),
            )
            movies.append(movie)

    if not movies:
        raise ValueError("Dataset is empty or contains no valid genre entries.")

    return movies


def build_genre_vector(genres: List[str]) -> List[float]:
    """
    Convert a list of genres into a binary feature vector.

    Each position corresponds to one genre in VALID_GENRES.
    1.0 means the movie has that genre; 0.0 means it does not.
    """
    return [1.0 if genre in genres else 0.0 for genre in VALID_GENRES]


def build_user_preference_vector(favorite_genre: str) -> List[float]:
    """
    Build a user preference vector from the selected favorite genre.

    For this beginner model, the user vector is a one-hot encoding
    of their favorite genre.
    """
    if favorite_genre not in VALID_GENRES:
        raise ValueError(f"Invalid genre: {favorite_genre}")

    return build_genre_vector([favorite_genre])


def cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    """
    Calculate cosine similarity between two genre vectors.

    Formula:
        similarity = (A · B) / (||A|| * ||B||)

    Returns a value between 0.0 and 1.0 for binary genre vectors.
    """
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    # Avoid division by zero if a vector has no active features
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


def compute_recommendations(
    movies: List[Movie],
    user_vector: List[float],
    top_n: int = TOP_N,
) -> List[Tuple[Movie, float]]:
    """
    Rank movies by cosine similarity between user preference and movie genres.

    A small rating boost is applied so higher-rated movies break ties fairly.
    Final score = similarity + (rating / 100)
    """
    scored_movies: List[Tuple[Movie, float]] = []

    for movie in movies:
        similarity = cosine_similarity(user_vector, movie.genre_vector)

        # Tiny rating adjustment improves ranking among similar genre matches
        final_score = similarity + (movie.rating / 100.0)

        scored_movies.append((movie, final_score))

    # Sort by score descending, then by rating descending as a secondary key
    scored_movies.sort(key=lambda item: (item[1], item[0].rating), reverse=True)

    return scored_movies[:top_n]


def normalize_genre_input(user_input: str) -> str:
    """
    Normalize user genre input for flexible matching.

    Examples:
        "sci-fi" -> "Sci-Fi"
        "SCI FI" -> "Sci-Fi"
        "action" -> "Action"
    """
    cleaned = user_input.strip().lower()

    aliases = {
        "sci fi": "Sci-Fi",
        "sci-fi": "Sci-Fi",
        "scifi": "Sci-Fi",
        "science fiction": "Sci-Fi",
    }

    if cleaned in aliases:
        return aliases[cleaned]

    for genre in VALID_GENRES:
        if cleaned == genre.lower():
            return genre

    return user_input.strip()


def get_favorite_genre() -> str:
    """
    Prompt the user for a favorite genre with validation and retry logic.
    """
    print("\nAvailable genres:")
    for index, genre in enumerate(VALID_GENRES, start=1):
        print(f"  {index}. {genre}")

    while True:
        user_input = input("\nEnter your favorite genre (name or number): ").strip()

        if not user_input:
            print("Error: Input cannot be empty. Please try again.")
            continue

        # Allow numeric selection (e.g., "3" for Horror)
        if user_input.isdigit():
            choice = int(user_input)
            if 1 <= choice <= len(VALID_GENRES):
                return VALID_GENRES[choice - 1]
            print(f"Error: Please enter a number between 1 and {len(VALID_GENRES)}.")
            continue

        normalized = normalize_genre_input(user_input)
        if normalized in VALID_GENRES:
            return normalized

        print(
            "Error: Invalid genre. Please choose from: "
            + ", ".join(VALID_GENRES)
        )


def display_banner() -> None:
    """Print a simple welcome banner for the CLI."""
    print("=" * 60)
    print("        MOVIE RECOMMENDATION SYSTEM")
    print("   Content-Based Genre Similarity Engine")
    print("=" * 60)


def display_recommendations(
    favorite_genre: str,
    recommendations: List[Tuple[Movie, float]],
) -> None:
    """Display ranked movie recommendations in a readable table format."""
    print(f"\nTop recommendations for fans of {favorite_genre}:\n")
    print(f"{'Rank':<6}{'Title':<28}{'Year':<8}{'Rating':<8}{'Score':<8}{'Genres'}")
    print("-" * 78)

    for rank, (movie, score) in enumerate(recommendations, start=1):
        genre_text = ", ".join(movie.genres)
        print(
            f"{rank:<6}{movie.title:<28}{movie.year:<8}{movie.rating:<8.1f}"
            f"{score:<8.3f}{genre_text}"
        )

    print("\nSimilarity score combines genre match (cosine similarity) with rating boost.")


def main() -> None:
    """Main entry point for the command-line recommendation system."""
    display_banner()

    try:
        movies = load_movies(DATASET_PATH)
    except (FileNotFoundError, ValueError) as error:
        print(f"\nFailed to load dataset: {error}")
        sys.exit(1)

    print(f"\nLoaded {len(movies)} movies from dataset.")

    favorite_genre = get_favorite_genre()
    user_vector = build_user_preference_vector(favorite_genre)
    recommendations = compute_recommendations(movies, user_vector)

    # Filter out zero-similarity results (no genre overlap at all)
    recommendations = [(movie, score) for movie, score in recommendations if score > 0]

    if not recommendations:
        print(
            f"\nNo matching movies found for '{favorite_genre}'. "
            "Try another genre."
        )
        sys.exit(0)

    display_recommendations(favorite_genre, recommendations)


if __name__ == "__main__":
    main()
