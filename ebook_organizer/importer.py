# importer.py
import pandas as pd
from sqlalchemy import create_engine, text
import ast
import numpy as np

# --- 1. PLEASE CONFIRM THIS LINE ---
# Since you are on Postgres.app, your username is your Mac's username 
# (which I'm guessing is 'aparajitapallav') and there is no password.
# Format: "postgresql://YOUR_MAC_USERNAME@localhost:5432/libris_ai"
DATABASE_URL = "postgresql://postgres:053122@localhost:5432/libris_ai"
# ----------------------------------

# Path to your Excel file (from our previous chats)
# EXCEL_FILE_PATH = "/Users/aparajitapallav/Desktop/Books/AI_Library_Report.xlsx"
EXCEL_FILE_PATH = "/Users/aparajitapallav/Google Drive/Books - Pallav/AI_Library_Report.xlsx"

engine = create_engine(DATABASE_URL)

print(f"Reading Excel file from: {EXCEL_FILE_PATH}")
# Read the file, replacing blank cells (NaN) with None
df = pd.read_excel(EXCEL_FILE_PATH).replace({np.nan: None})

print(f"Found {len(df)} books to import.")

with engine.connect() as connection:
    trans = connection.begin()
    try:
        connection.execute(text("DELETE FROM books"))
        print("Cleared existing books from the database.")
        
        books_imported = 0
        books_skipped = 0

        for index, row in df.iterrows():
            
            # --- THIS IS THE FIX ---
            # If the title is blank (None, NaN, etc.), skip this row
            if not row['title']:
                books_skipped += 1
                continue
            # --- END OF FIX ---

            # --- Process Authors ---
            # ... (rest of the loop is unchanged) ...
            authors_list = None
            if row['authors']:
                try:
                    authors_list = ast.literal_eval(row['authors'])
                except (ValueError, SyntaxError):
                    authors_list = [row['authors']]
            
            # --- Process Series Info ---
            series_name = None
            series_number = None
            if row['series_info']:
                try:
                    series_dict = ast.literal_eval(row['series_info'])
                    if isinstance(series_dict, dict):
                        series_name = series_dict.get('series_name')
                        series_number = series_dict.get('series_number')
                except (ValueError, SyntaxError):
                    pass 

            insert_query = text("""
                INSERT INTO books (title, authors, publication_year, genre, goodreads_rating, review_summary, series_name, series_number)
                VALUES (:title, :authors, :pub_year, :genre, :rating, :summary, :s_name, :s_num)
            """)
            
            connection.execute(insert_query, {
                "title": row['title'],
                "authors": authors_list,
                "pub_year": row['publication_year'],
                "genre": row['genre'],
                "rating": row['goodreads_rating'],
                "summary": row['review_summary'],
                "s_name": series_name,
                "s_num": series_number,
            })
            books_imported += 1 # Count the successful import
        
        trans.commit()
        print("✅ Successfully imported!")
        print(f"   {books_imported} books were added to the database.")
        print(f"   {books_skipped} blank rows were skipped.")
    
    except Exception as e:
        trans.rollback()
        print(f"❌ ERROR: An error occurred. Rolling back changes. {e}")