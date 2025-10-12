import pandas as pd
from pathlib import Path
import shutil
import re

def sanitize_filename(name: str) -> str:
    """Removes illegal characters from a string to make it a valid filename/directory name."""
    if not isinstance(name, str):
        name = str(name)
    # Remove characters that are invalid in Windows, macOS, and Linux file systems
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

def reorganize_library(df: pd.DataFrame, source_folder: str):
    """
    Reorganizes e-book files based on the enriched metadata in the DataFrame.

    This function is a generator that yields status updates for each file operation.

    Args:
        df: The DataFrame containing the book filepaths and enriched metadata.
        source_folder: The root folder where the reorganization will happen.
    """
    base_path = Path(source_folder)
    output_dir = base_path / "THE LIST"
    output_dir.mkdir(exist_ok=True)

    for index, row in df.iterrows():
        # Skip files if essential metadata like title or authors is missing
        if pd.isna(row.get('title')) or not row.get('authors'):
            yield f"⚠️ Skipping '{row['filename']}' due to missing essential metadata."
            continue

        try:
            # 1. Sanitize metadata for use in paths
            genre = sanitize_filename(row.get('genre', 'Uncategorized'))
            # Use only the first author for the folder name to keep paths cleaner
            author = sanitize_filename(row['authors'][0])
            title = sanitize_filename(row['title'])
            
            target_path = output_dir / genre / author
            
            # 2. Determine new filename and path based on series info
            series_info = row.get('series_info')
            if series_info and isinstance(series_info, dict) and series_info.get('series_name'):
                series_name = sanitize_filename(series_info['series_name'])
                series_number = series_info.get('series_number', 0)
                target_path = target_path / series_name
                
                # Format for series: "Series Name - 01 - Book Title.epub"
                new_filename = f"{series_name} - {series_number:02d} - {title}{row['extension']}"
            else:
                # Format for standalone books: "Book Title - Author.epub"
                new_filename = f"{title} - {author}{row['extension']}"

            # 3. Create the target directory if it doesn't exist
            target_path.mkdir(parents=True, exist_ok=True)
            
            # 4. Move and rename the file
            original_file = Path(row['filepath'])
            if original_file.exists():
                destination_file = target_path / new_filename
                shutil.move(original_file, destination_file)
                yield f"✅ Moved '{row['filename']}' to '{destination_file.relative_to(base_path)}'"
            else:
                yield f"⚠️ File '{row['filename']}' not found at original path. It might have been moved already."
            
        except Exception as e:
            yield f"❌ Error organizing '{row['filename']}': {e}"