import asyncio
import base64
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd
import streamlit as st
from textwrap import dedent

from metrics import ComprehensiveMetrics
from AIC import (
    AICommittee,
    hierarchical_deduplication,
    process_row_with_semaphore,
    setup_logger,
)


def generate_row_id(row: Dict[str, Any]) -> str:
    """Mirror the row id format used inside refactoredAIC for a consistent UI."""
    return f"Row_{hash(json.dumps(row, sort_keys=True)) % 10000:04d}"


def encode_logo_to_base64(path: Path) -> str:
    """Return a base64 data URI for an image so Streamlit can embed local files."""
    try:
        with path.open("rb") as image_file:
            b64_data = base64.b64encode(image_file.read()).decode("utf-8")
            return f"data:image/png;base64,{b64_data}"
    except FileNotFoundError:
        return ""


async def process_row_with_tracking(
    row_id: str,
    row: Dict[str, Any],
    committee: AICommittee,
    all_columns: List[str],
    semaphore: asyncio.Semaphore,
    status_callback: Callable[[str, str, str], None],
) -> Dict[str, Any]:
    """Process a row while emitting status updates for the UI."""
    try:
        status_callback(row_id, "Processing", "Agents validating data point…")
        result = await process_row_with_semaphore(committee, row, all_columns, semaphore)
    except Exception as exc:
        status_callback(row_id, "ERROR", f"Unexpected failure: {exc}")
        return {"status": "ERROR", "reason": str(exc)}

    status = result.get("status", "UNKNOWN")
    detail = result.get("reason", "")

    if status == "ACCEPT":
        detail = "Accepted and aligned to schema."
    elif status == "DISCOVERED_NEW":
        discovered = len(result.get("data", [])) if isinstance(result.get("data"), list) else 0
        detail = f"Discovered {discovered} additional candidate(s)."
    elif status == "REJECT" and not detail:
        detail = "Rejected (see logs for details)."
    elif status.startswith("SKIP") and not detail:
        detail = "Skipped due to upstream error."

    status_callback(row_id, status, detail)
    return result


async def run_pipeline_ui(
    input_csv_path: str,
    output_csv_path: str,
    dataset_description: str,
    schema_fields: List[str],
    model_name: str,
    concurrency: int,
    price_input: float,
    price_output: float,
    status_callback: Callable[[str, str, str], None],
    progress_callback: Callable[[str], None],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str, str]:
    """Run the full AI Committee pipeline while streaming updates back to the UI."""
    df = pd.read_csv(input_csv_path)
    all_df_columns = df.columns.tolist()

    output_path = Path(output_csv_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix("").as_posix() + "_frontend.log"
    logger = setup_logger(log_path)

    metrics = ComprehensiveMetrics(price_input=price_input, price_output=price_output)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    committee = AICommittee(llm_model_name=model_name, metrics_tracker=metrics, logger=logger)
    await committee._initialize_framework(dataset_description, schema_fields, df)

    candidates = df.to_dict("records")
    accepted_results: List[Dict[str, Any]] = []
    processed_hashes = set()
    processed_counter = 0
    iteration = 1

    while candidates:
        unique_candidates = []
        for cand in candidates:
            cand_hash = hashlib.sha256(json.dumps(cand, sort_keys=True).encode()).hexdigest()
            if cand_hash not in processed_hashes:
                processed_hashes.add(cand_hash)
                unique_candidates.append(cand)

        if not unique_candidates:
            break

        progress_callback(f"Pass {iteration}: dispatching {len(unique_candidates)} data point(s)…")
        tasks = []
        for cand in unique_candidates:
            row_id = generate_row_id(cand)
            status_callback(row_id, "Queued", f"Pass {iteration} | awaiting agents…")
            tasks.append(
                asyncio.create_task(
                    process_row_with_tracking(
                        row_id=row_id,
                        row=cand,
                        committee=committee,
                        all_columns=all_df_columns,
                        semaphore=semaphore,
                        status_callback=status_callback,
                    )
                )
            )

        iteration += 1
        next_round_candidates: List[Dict[str, Any]] = []

        for task in asyncio.as_completed(tasks):
            result = await task
            processed_counter += 1
            progress_callback(f"Processed {processed_counter} data point(s)…")
            if not result:
                continue
            status = result.get("status")
            if status == "ACCEPT":
                accepted_results.append(result["data"])
            elif status == "DISCOVERED_NEW":
                next_round_candidates.extend(result.get("data", []))

        candidates = next_round_candidates

    deduplicated_results = hierarchical_deduplication(accepted_results, schema_fields)

    final_results: List[Dict[str, Any]] = []
    progress_callback("Running final integrity validation…")
    for dp in deduplicated_results:
        report = await committee.data_integrity_validator_agent(dp)
        if report and report.is_plausible:
            final_results.append(dp)

    await committee.content_manager.close()

    if final_results:
        final_df = pd.DataFrame(final_results, columns=all_df_columns)
        final_df.to_csv(output_path, index=False)

    metrics_summary = metrics.calculate_summary()
    return final_results, metrics_summary, str(output_path), log_path

# -----------------------------------------------------------------------------
# UI BUILDER WITH CUSTOM STYLING
# -----------------------------------------------------------------------------

def build_frontend():
    st.set_page_config(page_title="CHIP AI Committee", layout="wide", page_icon="🧬")

    # --- CSS Styling for Dark Blue Theme ---
    # The primary colors are set in .streamlit/config.toml
    # This CSS handles additional customizations
    st.markdown(
        """
        <style>
        /* Custom Header Container */
        .custom-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: transparent;
            padding: 0.5rem 0;
            border-bottom: none;
            margin-bottom: 1rem;
        }
        
        /* Logo Images */
        .header-logo {
            height: 70px;
            object-fit: contain;
        }

        /* Orange focus outline for ALL input elements */
        input:focus, textarea:focus, [tabindex]:focus {
            outline: 2px solid #ff8c00 !important;
            outline-offset: 0px !important;
            border-color: #ff8c00 !important;
        }
        
        /* Text inputs and text areas */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea {
            border: 1px solid #4a5e82 !important;
        }
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus {
            border-color: #ff8c00 !important;
            box-shadow: 0 0 0 1px #ff8c00 !important;
        }

        /* Selectbox - black background */
        .stSelectbox > div > div {
            background-color: #000000 !important;
            color: #ffffff !important;
        }
        .stSelectbox > div > div:focus-within {
            border-color: #ff8c00 !important;
            box-shadow: 0 0 0 1px #ff8c00 !important;
        }
        
        /* File Uploader - black background, white label text */
        .stFileUploader > div {
            background-color: #000000 !important;
            border-radius: 5px;
            padding: 10px;
        }
        .stFileUploader label,
        .stFileUploader p,
        .stFileUploader span,
        .stFileUploader div {
            color: #ffffff !important;
        }
        
        /* Buttons - black background */
        .stButton > button,
        .stFormSubmitButton > button,
        .stDownloadButton > button {
            background-color: #000000 !important;
            color: #ffffff !important;
            border: 1px solid #000000 !important;
        }
        .stButton > button:hover,
        .stFormSubmitButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: #ffffff !important;
            background-color: #222222 !important;
        }
        .stButton > button:focus,
        .stFormSubmitButton > button:focus {
            box-shadow: 0 0 0 2px #ff8c00 !important;
        }

        /* Slider - orange thumb and track */
        .stSlider > div > div > div > div {
            background-color: #ff8c00 !important;
        }
        .stSlider [role="slider"] {
            background-color: #ff8c00 !important;
            border-color: #ff8c00 !important;
        }
        .stSlider > div > div > div:first-child {
            background: linear-gradient(to right, #ff8c00 var(--value-percent, 50%), #3a4563 var(--value-percent, 50%)) !important;
        }

        /* Expander - black background */
        .stExpander {
            border: 1px solid #4a5e82 !important;
            border-radius: 5px;
        }
        .stExpander > details > summary {
            background-color: #000000 !important;
            color: #ffffff !important;
            padding: 0.75rem 1rem;
            border-radius: 5px;
        }
        .stExpander > details[open] > summary {
            background-color: #000000 !important;
            color: #ffffff !important;
            border-radius: 5px 5px 0 0;
        }
        .stExpander > details > div {
            background-color: #0e1a35 !important;
        }
        
        /* Tables (Dataframes) */
        .stDataFrame {
            background-color: #0e1a35;
        }

        /* Hide standard Streamlit header for cleaner look */
        header[data-testid="stHeader"] {
            background-color: rgba(0,0,0,0);
        }
        
        /* Number input styling */
        .stNumberInput > div > div > input {
            border: 1px solid #4a5e82 !important;
        }
        .stNumberInput > div > div > input:focus {
            border-color: #ff8c00 !important;
            box-shadow: 0 0 0 1px #ff8c00 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    chip_logo_path = Path("chip.png")
    hms_logo_path = Path("hms.png")
    bch_logo_path = Path("bch.png")

    chip_logo_src = encode_logo_to_base64(chip_logo_path)
    hms_logo_src = encode_logo_to_base64(hms_logo_path)
    bch_logo_src = encode_logo_to_base64(bch_logo_path)

    if chip_logo_src:
        chip_section = f"<img src='{chip_logo_src}' class='header-logo' alt='CHIP Logo'>"
    else:
        chip_section = (
            "<h1 style='margin:0; font-size: 3rem; font-weight: 300; letter-spacing: -2px;'>"
            "ch<span style='font-weight:bold'>!</span>p</h1>"
            "<div style='font-size: 0.8rem; line-height: 1.2; opacity: 0.9;'>"
            "Computational<br>Health<br>Informatics<br>Program</div>"
        )

    if hms_logo_src:
        hms_section = (
            f"<img src='{hms_logo_src}' class='header-logo' style='height: 80px;' "
            "alt='Harvard Medical School'>"
        )
    else:
        hms_section = "<span style='font-size:1.2rem;'>Harvard Medical School</span>"

    if bch_logo_src:
        bch_section = (
            f"<img src='{bch_logo_src}' class='header-logo' alt='Boston Children&#39;s Hospital'>"
        )
    else:
        bch_section = "<span style='font-size:1.2rem;'>Boston Children&apos;s Hospital</span>"

    header_html = dedent(
        f"""
        <div class="custom-header">
            <div style="flex: 1; text-align: left;">{chip_section}</div>
            <div style="flex: 1; text-align: center;">{hms_section}</div>
            <div style="flex: 1; text-align: right;">{bch_section}</div>
        </div>
        """
    )

    st.markdown(header_html, unsafe_allow_html=True)

    st.markdown(
        "<h1 style='text-align: center; color: #ffffff; margin-top: 0.5rem; margin-bottom: 1.5rem; "
        "font-size: 2.5rem; font-weight: 600; letter-spacing: 1px;'>The AI Committee</h1>",
        unsafe_allow_html=True,
    )

    st.write(
        "Upload a dataset, provide your OpenAI credentials, and watch the AI Committee "
        "verify each data point in real time."
    )

    with st.expander("How it works", expanded=False):
        st.markdown(
            """
            - **Input**: The uploader accepts CSV files.
            - **Schema**: Fields should match the dataset columns the AI Committee expects.
            - **Security**: The OpenAI API key never leaves this session.
            - **Output**: Results and logs are written to the local output path.
            """
        )

    form = st.form("config_form", clear_on_submit=False)
    
    # Layout the form inputs
    c1, c2 = form.columns(2)
    with c1:
        uploaded_csv = st.file_uploader("Dataset CSV", type=["csv"])
        dataset_description = st.text_area("Dataset Description", height=100, placeholder="Describe the data context...")
    with c2:
        schema_fields_entry = st.text_input(
            "Schema Fields", help="Comma separated list (e.g.: name, age, diagnosis)"
        )
        default_output = str(Path.cwd() / "results" / "validated_output.csv")
        output_path = st.text_input("Output CSV Path", value=default_output)
        api_key = st.text_input("OpenAI API Key", type="password")

    c3, c4 = form.columns(2)
    with c3:
        model_choice = st.selectbox(
            "Model",
            options=["gpt-4o-mini", "o4-mini", "GPT-5"],
            index=0,
            help="Only these three models are tested; pricing is fixed per model.",
        )
    with c4:
        concurrency = st.slider("Max Concurrent Rows", min_value=1, max_value=10, value=2)

    # Fixed pricing per model (USD per 1M tokens), not user-editable.
    model_pricing = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "o4-mini": {"input": 1.00, "output": 4.40},
        "GPT-5": {"input": 1.25, "output": 10.00},
    }
    selected_pricing = model_pricing[model_choice]
    price_input = selected_pricing["input"] / 1_000_000
    price_output = selected_pricing["output"] / 1_000_000

    with form.expander("Pricing (fixed per model)"):
        st.write(
            f"{model_choice} pricing is fixed at "
            f"${selected_pricing['input']:.2f} per 1M input tokens "
            f"and ${selected_pricing['output']:.2f} per 1M output tokens."
        )


    submitted = form.form_submit_button("Run Evaluation", use_container_width=True)

    status_placeholder = st.empty()
    progress_placeholder = st.empty()
    metrics_placeholder = st.empty()
    results_placeholder = st.empty()

    if not submitted:
        return

    # --- Validation ---
    if not uploaded_csv:
        st.error("Please upload a CSV dataset.")
        return
    if not dataset_description.strip():
        st.error("Dataset description is required.")
        return
    if not schema_fields_entry.strip():
        st.error("Please provide schema fields.")
        return
    if not api_key.strip():
        st.error("OpenAI API key is required.")
        return

    schema_fields = [field.strip() for field in schema_fields_entry.replace(",", " ").split() if field.strip()]

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_csv.getvalue())
        input_csv_path = tmp.name

    os.environ["OPENAI_API_KEY"] = api_key.strip()

    # --- Tracking Callbacks ---
    row_status_map: Dict[str, Dict[str, str]] = {}

    def update_status(row_id: str, status: str, detail: str):
        row_status_map[row_id] = {"Row ID": row_id, "Status": status, "Details": detail}
        status_df = pd.DataFrame(row_status_map.values())
        # Re-render status table
        status_placeholder.dataframe(
            status_df.set_index("Row ID"),
            use_container_width=True,
            height=300
        )

    def update_progress(message: str):
        progress_placeholder.info(message)

    # --- Execution ---
    try:
        update_progress("Initializing AI Committee...")
        with st.spinner("Processing..."):
            final_results, metrics_summary, saved_output, log_path = asyncio.run(
                run_pipeline_ui(
                    input_csv_path=input_csv_path,
                    output_csv_path=output_path,
                    dataset_description=dataset_description.strip(),
                    schema_fields=schema_fields,
                    model_name=model_choice,
                    concurrency=concurrency,
                    price_input=price_input,
                    price_output=price_output,
                    status_callback=update_status,
                    progress_callback=update_progress,
                )
            )
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        return
    finally:
        Path(input_csv_path).unlink(missing_ok=True)

    # --- Results Display ---
    if final_results:
        results_placeholder.success(f"Saved {len(final_results)} validated row(s) to {saved_output}")
        st.dataframe(pd.DataFrame(final_results), use_container_width=True)
    else:
        results_placeholder.warning("No rows passed the final validation stage.")

    if metrics_summary:
        with metrics_placeholder.container():
            st.subheader("Metrics Summary")
            cost_summary = metrics_summary.get("cost_and_api_usage", {})
            total_cost = cost_summary.get("total_cost ($)", 0.0)
            total_input_tokens = cost_summary.get("total_prompt_tokens", 0)
            total_output_tokens = cost_summary.get("total_completion_tokens", 0)

            c_m1, c_m2, c_m3 = st.columns(3)
            c_m1.metric("Total Cost", f"${total_cost:.4f}")
            c_m2.metric("Input Tokens", f"{total_input_tokens:,}")
            c_m3.metric("Output Tokens", f"{total_output_tokens:,}")
    else:
        metrics_placeholder.info("Metrics summary unavailable.")

    st.caption(f"Detailed logs written to: {log_path}")


if __name__ == "__main__":
    build_frontend()