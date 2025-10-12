from pathlib import Path
import pandas as pd

def find_ebooks(source_folder: str) -> pd.DataFrame:
    """
    Recursively scans a directory for .epub and .pdf files.

    Args:
        source_folder: The absolute path to the folder to scan.

    Returns:
        A pandas DataFrame with columns for 'filepath', 'filename', and 'extension'.
        Returns an empty DataFrame if no files are found.
    
    Raises:
        ValueError: If the provided path is not a valid directory.
    """
    source_path = Path(source_folder)
    
    if not source_path.is_dir():
        raise ValueError(f"Error: The provided path '{source_folder}' is not a valid directory.")

    # Define the file extensions we are looking for
    supported_extensions = ['.epub','.EPUB', '.pdf']
    
    # Use rglob for recursive search and create a list of Path objects
    ebook_files = [
        file_path for ext in supported_extensions 
        for file_path in source_path.rglob(f'*{ext}')
    ]

    if not ebook_files:
        return pd.DataFrame()

    # Create a dictionary to build the DataFrame
    data = {
        'filepath': [str(f) for f in ebook_files],
        'filename': [f.name for f in ebook_files],
        'extension': [f.suffix for f in ebook_files]
    }
    
    df = pd.DataFrame(data)
    
    print(f"🔍 Found {len(df)} e-books in '{source_folder}'.")
    return df