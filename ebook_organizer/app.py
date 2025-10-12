import gradio as gr
import pandas as pd
import time
from pathlib import Path
import os  # <-- ADD THIS IMPORT

# Import our custom modules
from file_scanner import find_ebooks
from ai_agents import get_book_metadata
from file_organizer import reorganize_library
from data_models import BookMetadata  # <-- FIX THE IMPORT PATH HERE

# --- Main Processing Function ---
def process_library(folder_path: str):
    """
    The main orchestrator function triggered by the Gradio interface.
    It scans, enriches, reports, and organizes the e-book library.
    """
    if not folder_path:
        yield "❌ Please provide a folder path."
        return

    log = []
    
    # === PHASE 1: FILE DISCOVERY ===
    log.append("🔎 **Phase 1: Scanning for e-books...**")
    yield "\n".join(log)
    try:
        df = find_ebooks(folder_path)
        if df.empty:
            log.append("No e-book files (.epub, .pdf) found. Nothing to do.")
            yield "\n".join(log)
            return
        log.append(f"📚 Found {len(df)} e-books. Starting metadata enrichment...")
        yield "\n".join(log)
    except ValueError as e:
        yield f"❌ **Error:** {e}"
        return

    # Initialize columns for enriched data
    enriched_cols = list(BookMetadata.model_fields.keys())
    for col in enriched_cols:
        df[col] = pd.Series(dtype='object') 

    # === PHASE 2: AI-POWERED METADATA ENRICHMENT ===
    log.append("\n🧠 **Phase 2: Researching metadata with AI Agent...**")
    yield "\n".join(log)
    
    total_books = len(df)
    for index, row in df.iterrows():
        filename = row['filename']
        log.append(f"\n({index + 1}/{total_books}) Researching: **{filename}**")
        yield "\n".join(log)

        try:
            # CORRECTLY UNPACK THE 3 RETURN VALUES
            metadata, raw_result, crew_logs = get_book_metadata(filename)
            
            # Display the captured logs in the UI
            if crew_logs:
                log.append(f"🕵️‍♂️ **CrewAI Trace:**\n```\n{crew_logs}\n```")
                yield "\n".join(log)

            if metadata:
                for col in enriched_cols:
                    df.loc[index, col] = getattr(metadata, col)
                log.append(f"   👍 Success: Found metadata for '{metadata.title}'")
            else:
                log.append(f"   ⚠️ Failed to parse AI output for {filename}. Skipping.")
                log.append(f"   Raw output: `{raw_result}`")

            yield "\n".join(log)

        except Exception as e:
            log.append(f"   ❌ An unexpected error occurred for {filename}: {e}. Skipping.")
            yield "\n".join(log)
            time.sleep(1)

    # === PHASE 3: REPORTING ===
    log.append("\n📊 **Phase 3: Generating Excel report...**")
    yield "\n".join(log)
    report_path = Path(folder_path) / "AI_Library_Report.xlsx"
    df.to_excel(report_path, index=False)
    log.append(f"   ✅ Report saved to `{report_path}`")
    yield "\n".join(log)
    
    # === PHASE 4: INTELLIGENT FILE REORGANIZATION ===
    log.append("\n🗂️ **Phase 4: Organizing files into 'THE LIST' directory...**")
    yield "\n".join(log)
    
    organizer_generator = reorganize_library(df, folder_path)
    for org_log in organizer_generator:
        log.append(f"   {org_log}")
        yield "\n".join(log)

    # === PHASE 5: USER NOTIFICATION ===
    log.append("\n🎉 **Success! Your library has been organized.**")
    yield "\n".join(log)


# --- Gradio User Interface ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown("# 🤖 AI E-Book Library Organizer")
    gr.Markdown(
        "This tool uses an AI agent to automatically enrich your e-book metadata "
        "and organize the files into a clean `Genre/Author/Series` folder structure."
    )
    
    with gr.Row():
        folder_input = gr.Textbox(
            label="Enter Your E-Book Folder Path",
            placeholder="e.g., C:/Users/YourName/Documents/MyBooks"
        )
    
    start_button = gr.Button("Start Organizing", variant="primary")
    
    with gr.Accordion("Live Log", open=True):
        log_output = gr.Markdown("⏳ Awaiting instructions...")

    start_button.click(
        fn=process_library,
        inputs=[folder_input],
        outputs=[log_output]
    )

# --- Main Execution Block ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: OPENAI_API_KEY not found in .env file.  !!!")
        print("!!! The application will not work without it.        !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    demo.launch(inbrowser=True)