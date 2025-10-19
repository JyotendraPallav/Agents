# backend.py
from fastapi import FastAPI
from sqlalchemy import create_engine, text
from fastapi.middleware.cors import CORSMiddleware

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
@app.get("/api/books")
def get_all_books():
    """
    This is our API endpoint. When a browser requests this URL,
    this function will run.
    """
    print("Received request for /api/books")
    with engine.connect() as connection:
        # Run a SQL query to get all books, ordered by title
        result = connection.execute(text("SELECT * FROM books ORDER BY title"))

        # Convert the database rows into a list of dictionaries
        books = [dict(row._mapping) for row in result]

        # Send the list of books back as JSON
        return {"books": books}