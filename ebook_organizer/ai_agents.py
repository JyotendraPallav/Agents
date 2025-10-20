import os
import json
from crewai import Agent, Task, Crew
# from crewai_tools import DuckDuckGoSearchRunTool
from langchain_community.tools import DuckDuckGoSearchRun	
from langchain_community.callbacks import get_openai_callback
from pydantic import ValidationError
from crewai_tools import SerperDevTool
from langchain.chat_models import ChatOpenAI
import io
import contextlib
import re

# Import our Pydantic model
from data_models import BookMetadata
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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
    tools=[search_tool],
    llm=llm
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
            "'title', 'authors', 'publication_year', 'genre', 'goodreads_rating', 'review_summary', 'series_info' and 'image_url'.\n\n"
            "Key instructions:\n"
            "- 'authors' must be a list of strings.\n"
            "- 'series_info' must be an object with 'series_name' (string) and 'series_number' (integer).\n"
            # # --- NEW, STRATEGIC IMAGE INSTRUCTION ---
            # "**Image URL Strategy:**"
            # "1. First, search for the book's Wikipedia page (e.g., 'Ender's Game Wikipedia')."
            # "2. Find the primary book cover image on that page. It is the most reliable and public-facing image."
            # "3. The URL **MUST** end in .jpg, .jpeg, or .png. (e.g., a link to 'upload.wikimedia.org/.../...jpg')."
            # "4. **CRITICAL: Do NOT use any links from 'gr-assets.com' or 'goodreads.com'. They are BANNED and will be rejected.**"
            # "5. If you cannot find a valid Wikipedia or publisher image, set 'image_url' to null.\n"
            # # --- END OF CHANGE ---
            # --- NEW, FOCUSED ISBN INSTRUCTION ---
            "**- 'isbn': This is the MOST IMPORTANT field.** You must find the 13-digit ISBN (e.g., 9780590353427). "
            "If you can only find a 10-digit ISBN, that is also acceptable. "
            "If you cannot find any ISBN, set the value to null.\n"
            # --- END OF CHANGE ---
            "- If the book is not part of a series, 'series_info' should be null.\n"
            "- If any other piece of information cannot be reliably found, its value should be null.\n"
            "- Do not invent or guess information. Accuracy is paramount."
        ),
        expected_output="A single JSON object containing the book's verified metadata.",
        agent=researcher_agent
    )

# 3. Create a function to run the crew and validate the output

def get_book_metadata(filename: str) -> (BookMetadata, str, dict): # Returns 3 items now
    """
    Runs the CrewAI agent to get metadata for a book and validates the output.
    Returns the metadata, raw result, and token usage information.
    """
    book_crew = Crew(
        agents=[researcher_agent],
        tasks=[create_research_task(filename)],
        verbose=True
    )
    
    kickoff_output = None
    token_usage = {}

    try:
        # Wrap the kickoff call with the OpenAI callback handler
        with get_openai_callback() as cb:
            kickoff_output = book_crew.kickoff()
            # The callback object 'cb' now contains the token usage details
            token_usage = {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost": cb.total_cost,
            }

    except Exception as e:
        print(f"\nCRITICAL ERROR during crew.kickoff() for file: {filename}\nERROR: {e}\n")
        return None, "", {} # Return empty dict on failure

    raw_result = kickoff_output.raw if kickoff_output else ""

    if not raw_result:
        print(f"WARNING: Crew kickoff for {filename} produced no output.")
        return None, "", token_usage

    try:
        json_start_index = raw_result.find('{')
        json_end_index = raw_result.rfind('}') + 1

        if json_start_index != -1 and json_end_index != -1:
            json_str = raw_result[json_start_index:json_end_index]
            metadata = BookMetadata.model_validate_json(json_str)
            # Return all three values on success
            return metadata, raw_result, token_usage
        else:
            raise ValueError("No JSON object found in the agent's output.")
        
    except (ValidationError, json.JSONDecodeError, ValueError) as e:
        print(f"Validation/Parsing Error for {filename}: {e}\nRaw output: {raw_result}")
        return None, raw_result, token_usage