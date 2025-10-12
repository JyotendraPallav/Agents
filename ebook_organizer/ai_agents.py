import os
import json
from crewai import Agent, Task, Crew
# from crewai_tools import DuckDuckGoSearchRunTool
from langchain_community.tools import DuckDuckGoSearchRun	
from pydantic import ValidationError
from crewai_tools import SerperDevTool
import io
import contextlib
import re

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

# def get_book_metadata(filename: str) -> (BookMetadata, str, str):
#     """
#     Runs the CrewAI agent, validates the output, and generates a user-friendly log.
#     ...
#     Returns:
#         A tuple containing:
#         - The validated BookMetadata object (or None on failure).
#         - The raw JSON string result from the agent.
#         - A user-friendly summary of the crew's execution.
#     """
#     book_crew = Crew(
#         agents=[researcher_agent],
#         tasks=[create_research_task(filename)],
#         verbose=True
#     )
    
#     kickoff_output = None
#     captured_logs = ""
#     user_friendly_log = []

#     try:
#         log_stream = io.StringIO()
#         with contextlib.redirect_stdout(log_stream):
#             kickoff_output = book_crew.kickoff()
        
#         captured_logs = log_stream.getvalue()
#         print(captured_logs) # Keep printing full log to terminal for debugging

#         # --- NEW: Intelligent Log Parsing ---
#         if "Entering new CrewAgentExecutor chain" in captured_logs:
#             user_friendly_log.append("   ▶️ Agent activated: Starting research...")
        
#         # Find all search queries the agent made
#         searches = re.findall(r'\"search_query\": \"(.*?)\"', captured_logs)
#         if searches:
#             user_friendly_log.append(f"   🔎 Performing web search for: '{searches[0]}'")
#             if len(searches) > 1:
#                 user_friendly_log.append(f"   ...and {len(searches)-1} other searches.")
        
#         if "Finished chain." in captured_logs:
#             user_friendly_log.append("   ✅ Agent finished: Task completed.")

#     except Exception as e:
#         print(f"A CRITICAL ERROR occurred during crew.kickoff() for file: {filename}\nERROR: {e}")
#         return None, "", f"   ❌ Agent failed during execution: {e}"

#     raw_result = kickoff_output.raw if kickoff_output else ""

#     if not raw_result:
#         print(f"WARNING: Crew kickoff for {filename} produced no output.")
#         return None, "", "   ⚠️ Agent produced no output."

#     try:
#         # --- ROBUST JSON EXTRACTION ---
#         json_start_index = raw_result.find('{')
#         json_end_index = raw_result.rfind('}') + 1

#         if json_start_index != -1 and json_end_index != -1:
#             json_str = raw_result[json_start_index:json_end_index]
#             metadata = BookMetadata.model_validate_json(json_str)
            
#             # Add extracted info to the user-friendly log
#             user_friendly_log.append(f"   📄 Extracted: '{metadata.title}' by {', '.join(metadata.authors)}")
            
#             return metadata, json_str, "\n".join(user_friendly_log)
#         else:
#             raise ValueError("No JSON object found in the agent's output.")
        
#     except (ValidationError, json.JSONDecodeError, ValueError) as e:
#         error_message = f"   ❌ Failed to parse agent's final answer: {e}"
#         print(f"Validation/Parsing Error for {filename}: {e}\nRaw output: {raw_result}")
#         user_friendly_log.append(error_message)
#         return None, raw_result, "\n".join(user_friendly_log)

def get_book_metadata(filename: str) -> (BookMetadata, str): # <-- Note: Returns 2 items now
    """
    Runs the CrewAI agent to get metadata for a book and validates the output.
    Logs will now appear in real-time in the terminal.
    """
    book_crew = Crew(
        agents=[researcher_agent],
        tasks=[create_research_task(filename)],
        verbose=True  # This will now print directly to the terminal in real-time
    )
    
    kickoff_output = None
    try:
        # Call kickoff directly. Logs will stream to the console.
        kickoff_output = book_crew.kickoff()
    except Exception as e:
        print(f"\nCRITICAL ERROR during crew.kickoff() for file: {filename}\nERROR: {e}\n")
        return None, ""

    raw_result = kickoff_output.raw if kickoff_output else ""

    if not raw_result:
        print(f"WARNING: Crew kickoff for {filename} produced no output.")
        return None, ""

    try:
        # Robust JSON extraction
        json_start_index = raw_result.find('{')
        json_end_index = raw_result.rfind('}') + 1

        if json_start_index != -1 and json_end_index != -1:
            json_str = raw_result[json_start_index:json_end_index]
            metadata = BookMetadata.model_validate_json(json_str)
            # Return the metadata and the raw result for debugging in the UI
            return metadata, raw_result
        else:
            raise ValueError("No JSON object found in the agent's output.")
        
    except (ValidationError, json.JSONDecodeError, ValueError) as e:
        print(f"Validation/Parsing Error for {filename}: {e}\nRaw output: {raw_result}")
        return None, raw_result