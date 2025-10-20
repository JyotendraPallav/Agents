# backfiller.py
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import requests  # For making web requests to Google Books API

# Import the upgraded agent function
from ai_agents import get_book_metadata

def run_backfill():
    """
    Finds all books missing an image URL.
    - If ISBN is present, it directly fetches the image from Google Books.
    - If ISBN is missing, it uses the AI agent to find it first.
    """
    # 1. Load API Keys
    print("Loading API keys...")
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GOOGLE_BOOKS_API_KEY = os.getenv("GOOGLE_BOOKS_API_KEY")

    if not OPENAI_API_KEY:
        print("!!! ERROR: OPENAI_API_KEY not found. Please check your .env file. !!!")
        return
    if not GOOGLE_BOOKS_API_KEY:
        print("!!! ERROR: GOOGLE_BOOKS_API_KEY not found. Please check your .env file. !!!")
        return

    # 2. Connect to the Database
    DATABASE_URL = "postgresql://postgres:053122@localhost:5432/libris_ai"
    
    try:
        engine = create_engine(DATABASE_URL)
        
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            
            # 3. Find all books that are missing an image
            print("Fetching all books that are missing an image URL...")
            select_query = text("SELECT id, title, isbn FROM books WHERE image_url IS NULL")
            books_to_update = connection.execute(select_query).mappings().all()

            total_books = len(books_to_update)
            if total_books == 0:
                print("✅ All books already have an image URL. Nothing to do!")
                return
                
            print(f"Found {total_books} books to process. Starting...")
            print("---")

            # 4. Loop through each book
            for index, book in enumerate(books_to_update):
                print(f"\n({index + 1}/{total_books}) Processing: {book['title']}...")
                book_isbn = book['isbn']
                
                try:
                    # --- THIS IS THE NEW SMART LOGIC ---
                    if not book_isbn:
                        # 5a. ISBN is MISSING. We must run the AI agent.
                        print("   ISBN not found. Running AI agent...")
                        metadata, raw_result, token_usage = get_book_metadata(book['title'])
                        
                        if metadata and metadata.isbn:
                            book_isbn = metadata.isbn.replace("-", "")
                            print(f"   AI found ISBN: {book_isbn}.")
                            # We have the ISBN, now update it in the database
                            connection.execute(text("UPDATE books SET isbn = :isbn WHERE id = :id"), 
                                               {"isbn": book_isbn, "id": book['id']})
                        else:
                            print(f"   ⚠️ Agent did not find an ISBN for this book. Skipping.")
                            continue # Move to the next book
                    else:
                        # 5b. ISBN ALREADY EXISTS.
                        print(f"   ISBN already exists: {book_isbn}. Skipping AI agent.")
                    # --- END OF NEW LOGIC ---

                    # 6. If we have an ISBN (either new or old), call Google Books API
                    if book_isbn:
                        print(f"   Fetching image from Google Books API...")
                        google_api_url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{book_isbn}&key={GOOGLE_BOOKS_API_KEY}"
                        
                        try:
                            response = requests.get(google_api_url, timeout=5)
                            response.raise_for_status()
                            data = response.json()
                            
                            image_url = data.get("items", [{}])[0].get("volumeInfo", {}).get("imageLinks", {}).get("thumbnail")
                            
                            if image_url:
                                # 7. SUCCESS! Update the database
                                connection.execute(text("UPDATE books SET image_url = :image_url WHERE id = :id"), 
                                                   {"image_url": image_url, "id": book['id']})
                                print(f"   ✅ SUCCESS: Saved Google Books image URL.")
                            else:
                                print(f"   ⚠️ Google Books API had no image for this ISBN. Skipping.")

                        except requests.exceptions.RequestException as req_err:
                            print(f"   ⚠️ REJECTED: Could not call Google Books API (Error: {req_err}). Skipping.")
                
                except Exception as e:
                    print(f"   ❌ An unexpected error occurred for this book ({e}). Skipping.")
            
            print("\n🎉 Full backfill complete!")

    except Exception as e:
        print(f"❌ A critical error occurred: {e}")
        print("Please check your DATABASE_URL or network connection.")

if __name__ == "__main__":
    run_backfill()