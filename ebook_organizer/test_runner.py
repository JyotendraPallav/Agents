import pandas as pd
import os
from dotenv import load_dotenv

# Import our project modules
from file_scanner import find_ebooks
from ai_agents import get_book_metadata
from data_models import BookMetadata

def run_test():
    """
    A standalone test script to debug the DataFrame population issue.
    """
    # --- SETUP ---
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("!!! WARNING: OPENAI_API_KEY not found. The script will fail. !!!")
        return
        
    folder_path = "/Users/aparajitapallav/Desktop/Books"
    print(f"--- Starting Test for folder: {folder_path} ---")

    # --- PHASE 1: FILE DISCOVERY ---
    try:
        df = find_ebooks(folder_path)
        if df.empty:
            print("No e-book files found. Exiting test.")
            return
        print(f"Found {len(df)} e-books. Initial DataFrame:")
        print(df)
    except ValueError as e:
        print(f"Error during file scanning: {e}")
        return

    # Initialize columns for enriched data
    enriched_cols = list(BookMetadata.model_fields.keys())
    for col in enriched_cols:
        df[col] = None
    
    # Crucially, ensure the dtype allows for complex objects
    df = df.astype({col: 'object' for col in enriched_cols})


    # --- PHASE 2: METADATA ENRICHMENT (LOOP) ---
    for index, row in df.iterrows():
        filename = row['filename']
        print(f"\n--- Processing: {filename} (Index: {index}) ---")

        try:
            metadata, raw_result, user_friendly_log = get_book_metadata(filename)
            
            if metadata:
                # --- DEBUGGING PRINT STATEMENTS ---
                print("\n" + "="*50)
                print(f"DEBUGGING METADATA FOR: {filename}")
                print(f"  - Full Object: {metadata}")
                print(f"  - Authors: {metadata.authors} (Type: {type(metadata.authors)})")
                
                series_info_dict = None
                if metadata.series_info:
                    series_info_dict = metadata.series_info.model_dump()
                    print(f"  - Series Info: {series_info_dict} (Type: {type(series_info_dict)})")
                else:
                    print("  - Series Info: None")
                print("="*50 + "\n")
                
                # --- THE FIX: Using .at for robust cell assignment ---
                print("Attempting to populate DataFrame using .at indexer...")
                df.at[index, 'title'] = metadata.title
                df.at[index, 'authors'] = metadata.authors
                df.at[index, 'publication_year'] = metadata.publication_year
                df.at[index, 'genre'] = metadata.genre
                df.at[index, 'goodreads_rating'] = metadata.goodreads_rating
                df.at[index, 'review_summary'] = metadata.review_summary
                df.at[index, 'series_info'] = series_info_dict
                print("   ✅ Successfully populated DataFrame for this item.")
            
            else:
                print(f"   ⚠️ Agent failed to return valid metadata for {filename}.")

        except Exception as e:
            print(f"   ❌ A critical error occurred during processing for {filename}: {e}")
            # This will now show the real error without crashing the loop

    # --- FINAL RESULT ---
    print("\n\n--- Test Complete. Final DataFrame State ---")
    print(df)
    
    print("\n--- Displaying 'series_info' column specifically ---")
    print(df['series_info'])


if __name__ == "__main__":
    run_test()