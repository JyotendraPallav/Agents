# backend.py
from fastapi import FastAPI
from sqlalchemy import create_engine, text
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# --- IMPORTANT ---
# Use the SAME Database URL from importer.py
DATABASE_URL = "postgresql://postgres:053122@localhost:5432/libris_ai"
# ---------------
print(f"Connecting to database at: {DATABASE_URL}")
engine = create_engine(DATABASE_URL)
app = FastAPI()
print("FastAPI app initialized.")
# --- CORS Middleware ---
# This is a crucial security step. It tells your backend to
# trust the "frontend" (which will run on http://localhost:3000)
# and allow it to request data.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
print("CORS middleware added.")
# @app.get("/api/books")
# def get_all_books():
#     """
#     This is our API endpoint. When a browser requests this URL,
#     this function will run.
#     """
#     print("Received request for /api/books")
#     with engine.connect() as connection:
#         # Run a SQL query to get all books, ordered by title
#         result = connection.execute(text("SELECT * FROM books ORDER BY title"))

#         # Convert the database rows into a list of dictionaries
#         books = [dict(row._mapping) for row in result]

#         # Send the list of books back as JSON
#         return {"books": books}

# backend.py

# backend.py

@app.get("/api/books")
def get_all_books(search: Optional[str] = None, genre: Optional[str] = None):
    """
    Fetches all books, but can be filtered by a search term or genre.
    """
    print(f"Received request for /api/books (Search: {search}, Genre: {genre})")

    # Start building our SQL query and parameters
    query = "SELECT * FROM books"
    conditions = []
    params = {}

    if search:
        # Filter by title (case-insensitive)
        conditions.append("title ILIKE :search_term")
        params["search_term"] = f"%{search}%"

    if genre:
        # --- THIS IS THE FIX ---
        # Instead of an exact match, find if the genre string CONTAINS the word
        conditions.append("genre ILIKE :genre_term")
        params["genre_term"] = f"%{genre}%"
        # --- END OF FIX ---

    # If we have any filters, add the "WHERE" clause
    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY title"

    with engine.connect() as connection:
        result = connection.execute(text(query), params)
        books = [dict(row._mapping) for row in result]
        return {"books": books}


# --- ADD THIS NEW FUNCTION ---
@app.get("/api/books/{book_id}")
def get_single_book(book_id: int):
    """
    This endpoint fetches one specific book by its ID.
    """
    print(f"Received request for single book with ID: {book_id}")
    with engine.connect() as connection:
        # Use a "WHERE" clause to find the specific book
        query = text("SELECT * FROM books WHERE id = :id")
        result = connection.execute(query, {"id": book_id}).mappings().first()

        if result:
            return {"book": result}
        else:
            # Optional: Return a 404 error if the book isn't found
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail="Book not found")
# --- END OF NEW FUNCTION ---

# backend.py

@app.get("/api/genres")
def get_all_genres():
    """
    Fetches a list of all unique, non-null genres, splitting combined genres.
    """
    print("Received request for /api/genres")

    unique_genres = set()

    with engine.connect() as connection:
        # First, get all the messy genre strings from the database
        query = text("SELECT DISTINCT genre FROM books WHERE genre IS NOT NULL")
        combined_genre_list = connection.execute(query).mappings().all()

        for row in combined_genre_list:
            genre_string = row['genre']

            # Split the string by comma (e.g., "Horror, Fantasy")
            individual_genres = [g.strip() for g in genre_string.split(',')]

            # Add each clean, individual genre to our set
            unique_genres.update(individual_genres)

    # Convert the set (which has no duplicates) to a sorted list
    sorted_genres = sorted(list(unique_genres))

    return {"genres": sorted_genres}



# backend.py
