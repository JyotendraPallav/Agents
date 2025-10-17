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
    
    # --- NEW: Initialize total token and cost counters ---
    total_tokens_used = 0
    total_cost_incurred = 0.0
    
    # === PHASE 1: FILE DISCOVERY ===
    # ... (this part is unchanged) ...
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
    # for index, row in df.iterrows():
    #     filename = row['filename']
    #     log.append(f"\n({index + 1}/{total_books}) Researching: **{filename}**")
    #     yield "\n".join(log)
    for idx, (index, row) in enumerate(df.iterrows()):
        filename = row['filename']
        log.append(f"\n({idx + 1}/{total_books}) Researching: **{filename}**")
        yield "\n".join(log)

        try:
            # --- MODIFIED: Unpack the new 3-item return value ---
            metadata, raw_result, token_usage = get_book_metadata(filename)
            
            # --- NEW: Update and display token usage ---
            if token_usage:
                total_tokens_used += token_usage.get("total_tokens", 0)
                total_cost_incurred += token_usage.get("total_cost", 0.0)
                log.append(f"    Tokens Used: {token_usage.get('total_tokens', 0)}")
                log.append(f"   💰 **Total Cost So Far: ${total_cost_incurred:.4f}**")
                yield "\n".join(log)

            if metadata:
                log.append(f"   👍 Success: Extracted '{metadata.title}'")
                
                # Use .at for robustly setting values
                df.at[idx, 'title'] = metadata.title
                df.at[idx, 'authors'] = metadata.authors
                df.at[idx, 'publication_year'] = metadata.publication_year
                df.at[idx, 'genre'] = metadata.genre
                df.at[idx, 'goodreads_rating'] = metadata.goodreads_rating
                df.at[idx, 'review_summary'] = metadata.review_summary
                
                if metadata.series_info:
                    df.at[idx, 'series_info'] = metadata.series_info.model_dump()
                else:
                    df.at[idx, 'series_info'] = None
            else:
                log.append(f"   ⚠️ Agent failed to return valid metadata for {filename}.")

            yield "\n".join(log)

        except Exception as e:
            log.append(f"   ❌ An unexpected error occurred: {e}. Skipping.")
            yield "\n".join(log)
            time.sleep(1)

    # Final summary of token usage
    log.append("\n---")
    log.append(f"**Total Tokens Used for the Session:** {total_tokens_used}")
    log.append(f"**Estimated Total Cost:** ${total_cost_incurred:.2f}")


    # === PHASE 3: REPORTING ===
    log.append("\n📊 **Phase 3: Generating Excel report...**")
    yield "\n".join(log)
    report_path = Path(folder_path) / "AI_Library_Report.xlsx"
    
    # Create a copy for reporting to avoid changing the original df
    report_df = df.copy()

    # --- Format columns for human-readable output ---
    
    # Format 'authors' list into a single string: "Author A, Author B"
    report_df['authors'] = report_df['authors'].apply(
        lambda authors: ', '.join(authors) if isinstance(authors, list) else ''
    )

    # --- THIS IS THE MODIFIED PART ---
    # Format 'series_info' dict to only use the series name
    report_df['series_info'] = report_df['series_info'].apply(
        lambda info: info.get('series_name', '') if isinstance(info, dict) else ''
    )

    # Define the exact columns for the final report
    columns_for_report = [
        'title', 
        'authors', 
        'publication_year', 
        'genre', 
        'goodreads_rating', 
        'series_info',
        'review_summary' 
        # 'filename'
    ]
    
    # Save the formatted DataFrame to Excel
    report_df[columns_for_report].to_excel(report_path, index=False)
    
    log.append(f"   ✅ Report saved to `{report_path}`")
    yield "\n".join(log)
    
    # # === PHASE 4: INTELLIGENT FILE REORGANIZATION ===
    # log.append("\n🗂️ **Phase 4: Organizing files into 'THE LIST' directory...**")
    # yield "\n".join(log)
    
    # organizer_generator = reorganize_library(df, folder_path)
    # for org_log in organizer_generator:
    #     log.append(f"   {org_log}")
    #     yield "\n".join(log)

    # # === PHASE 5: USER NOTIFICATION ===
    # log.append("\n🎉 **Success! Your library has been organized.**")
    # yield "\n".join(log)


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