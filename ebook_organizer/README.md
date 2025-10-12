🤖 AI E-Book Library Organizer
An AI-powered desktop agent that automatically scans, enriches, and organizes your digital library of e-books. This tool uses a CrewAI agent to fetch rich metadata for your .epub and .pdf files and then physically reorganizes them into a clean, structured folder hierarchy.

---

## 📚 Features Overview

| Feature                        | Description                                                                                   |
| ------------------------------ | --------------------------------------------------------------------------------------------- |
| **Automated File Discovery**   | Recursively scans a source folder for `.epub` and `.pdf` files.                               |
| **AI Metadata Enrichment**     | Uses an AI agent to fetch: Title, Author(s), Year, Genre, Goodreads Rating, Review, Series.   |
| **Smart Organization**         | Moves/renames files into: `THE LIST/[Genre]/[Author]/[Series Name]/[Book File]`              |
| **Detailed Reporting**         | Exports a full Excel report (`AI_Library_Report.xlsx`) with enriched metadata.                |
| **Interactive Web UI**         | Gradio-based interface for folder selection, process control, and live logs.                  |

---

## 🛠️ Technology Stack

- **AI Agent Framework:** CrewAI ("Book Researcher" agent)
- **LLM:** OpenAI API (GPT-4o or similar)
- **Web UI:** Gradio
- **Data Handling:** Pandas
- **Core Logic:** Python 3.10+
- **Validation:** Pydantic
- **File System:** Pathlib

---

## ⚙️ Project Structure

```
ebook-organizer/
├── app.py              # Gradio UI entry point
├── file_scanner.py     # E-book file discovery
├── ai_agents.py        # CrewAI agent logic
├── file_organizer.py   # File moving/renaming
├── data_models.py      # Pydantic validation models
├── test_runner.py      # CLI test script
├── .env                # API key (create manually)
└── requirements.txt    # Dependencies
```

---

## 🚀 Getting Started

### 1. Prerequisites

- Python 3.10 or higher
- OpenAI API key

### 2. Installation

```bash
# Clone the repository and navigate to the project directory
cd path/to/ebook-organizer

# Create a virtual environment
python -m venv .venv
# Or using uv
uv venv

# Activate the virtual environment
# Windows:
# .\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# Or using uv
uv pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the project root with your OpenAI API key:

```
OPENAI_API_KEY="sk-..."
```

### 4. Running the Application

```bash
python app.py
```

- This launches the Gradio web server.
- Open your browser to the provided local URL (e.g., http://127.0.0.1:7860).

### 5. Usage

1. Enter the absolute path to your e-book folder in the UI.
2. Click **Start Organizing**.
3. Monitor progress in the **Live Log** (detailed logs in terminal).
4. On completion:
    - Organized files appear in a new `THE LIST` subfolder.
    - `AI_Library_Report.xlsx` is saved in your source folder.

---

## 💡 Tips

- Ensure your `.env` file is **not** committed to version control.
- For best results, use folders with only `.epub` and `.pdf` files.
- Review the Excel report for any missing or incorrect metadata.

---

## 📬 Feedback & Contributions

Feel free to open issues or submit pull requests to improve the project!

---