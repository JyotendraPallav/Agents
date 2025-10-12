import os
import json
from crewai import Agent, Task, Crew
# from crewai_tools import DuckDuckGoSearchRunTool
from langchain_community.tools import DuckDuckGoSearchRun	
from pydantic import ValidationError
from crewai_tools import SerperDevTool
import io
import contextlib

# Import our Pydantic model
from data_models import BookMetadata

# Initialize the search tool
search_tool = SerperDevTool()

# 1. Define the Researcher Agent
researcher_agent = Agent(
    role='Expert Literary Researcher',
    goal="Find and compile accurate, comprehensive metadata for books based on their filenames.",
    backstory=(
        "You are a world-class librarian and digital archivist. You have a knack for "
        "identifying books, even from messy filenames, and you meticulously structure your findings. "
        "You are precise and only return facts you can verify."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[search_tool]
)

# 2. Define the Research Task
def create_research_task(filename: str) -> Task:
    """Creates a research task for a given book filename."""
    return Task(
        description=(
            f"Analyze the book filename: '{filename}'. Your primary goal is to identify the book and "
            "extract its key metadata. Perform web searches, prioritizing sources like Goodreads, "
            "Amazon, and official publisher pages.\n\n"
            "Your final answer MUST BE a single, clean JSON object with the following keys: "
            "'title', 'authors', 'publication_year', 'genre', 'goodreads_rating', 'review_summary', 'series_info'.\n\n"
            "Key instructions:\n"
            "- 'authors' must be a list of strings.\n"
            "- 'series_info' must be an object with 'series_name' (string) and 'series_number' (integer).\n"
            "- If the book is not part of a series, 'series_info' should be null.\n"
            "- If any other piece of information cannot be reliably found, its value should be null.\n"
            "- Do not invent or guess information. Accuracy is paramount."
        ),
        expected_output="A single JSON object containing the book's verified metadata.",
        agent=researcher_agent
    )

# 3. Create a function to run the crew and validate the output
def get_book_metadata(filename: str) -> (BookMetadata, str, str):
    """
    Runs the CrewAI agent to get metadata for a book and validates the output.
    ...
    """
    book_crew = Crew(
        agents=[researcher_agent],
        tasks=[create_research_task(filename)],
        verbose=True
    )
    
    try:
        kickoff_output = book_crew.kickoff()
    except Exception as e:
        print(f"CRITICAL ERROR during crew kickoff: {e}")

    
    kickoff_output = None # Initialize to None

    # log_stream = io.StringIO()
    # with contextlib.redirect_stdout(log_stream):
    #     try:
    #         # Get the full CrewOutput object
    #         kickoff_output = book_crew.kickoff()
    #     except Exception as e:
    #         print(f"CRITICAL ERROR during crew kickoff: {e}")

    # captured_logs = log_stream.getvalue()
    # print(captured_logs) 
    # Extract the raw string from the output object IF it exists
    raw_result = kickoff_output.raw if kickoff_output else ""
    


    try:
        # The rest of the function now works with the extracted raw_result string
        if "```json" in raw_result:
            json_str = raw_result.split("```json\n")[1].split("```")[0]
        else:
            json_str = raw_result
        
        metadata = BookMetadata.model_validate_json(json_str)
        return metadata, json_str, captured_logs
    except (ValidationError, json.JSONDecodeError) as e:
        print(f"Pydantic Validation Error for {filename}: {e}\nRaw output: {raw_result}")
        return None, raw_result, captured_logs
    except Exception as e:
        print(f"An unexpected error occurred during metadata processing for {filename}: {e}")
        return None, raw_result, captured_logs