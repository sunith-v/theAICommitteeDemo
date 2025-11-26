## The AI Committee

The AI Committee is a multi‑agent validation pipeline for web data validation and remediation.
---

## 1. Environment setup

You can use either **conda**, or any other custom virtual environment. The following instructions will be in conda. Python 3.11 is recommended.


1. **Create and activate an environment**

```bash
conda create -n aic python=3.11 -y
conda activate aic
```

2. **Install Python dependencies**

From the project root:

```bash
pip install -r requirements.txt
```

---

## 2. Required configuration

The app uses the OpenAI API through `langchain-openai` and `openai`.

1. **Prepare an input CSV**

Your CSV should contain the columns you plan to validate. They will later be referenced in the UI as the **schema fields**. 

Your CSV also needs a "source" column: a URL column the system uses to crawl and fact‑check each row.

Place the CSV anywhere on disk; you will upload it via the UI.

---

## 3. Running the frontend (`frontend.py`)

The frontend is a Streamlit app defined in `frontend.py`. Run it with:

```bash
cd /Users/sunithv/Desktop/theAICommitee
streamlit run frontend.py
```

Streamlit will print a local URL, typically something like:

```text
  Local URL: http://localhost:8501
```

Open that URL in your browser to access the AI Committee UI.

---

## 4. Using the UI (how to operate it)

Once the Streamlit app is running:


1. **Upload your dataset**
   - In the left column, use **“Dataset CSV”** to upload a `.csv` file.
   - In **“Dataset Description”**, provide a one‑sentence description of what each row represents (e.g., “Each row is a news article about adverse drug events in US hospitals.”).  
     This description is used to contextualize the LLM agents.

2. **Define schema fields**
   - In the right column, fill in **“Schema Fields”** with the fields to validate, as a comma‑ or space‑separated list (e.g., `name, city, state, date`).
   - These names should match column names in the uploaded CSV.

3. **Choose output path & API key**
   - **“Output CSV Path”**: path on your machine where validated results will be written (defaults to a `results/validated_output.csv` path under the current working directory).
   - **“OpenAI API Key”**: paste your API key. It is only used locally in the current session.

4. **Model & concurrency**
   - **“Model”**: choose one of the three supported models: `gpt-4o-mini`, `o4-mini`, or `GPT-5`.
   - **“Max Concurrent Rows”**: controls how many rows are processed in parallel. Higher values may be faster but can increase API usage and rate‑limit risk.

5. **Pricing (fixed per model)**
   - Open the **“Pricing (fixed per model)”** expander to view the assumed token prices used for metrics.
   - For all three models, pricing is fixed at **$0.15 / 1M input tokens** and **$0.60 / 1M output tokens** and cannot be edited from the UI.

6. **Run the evaluation**
   - Click **“Run Evaluation”**.
   - The app will validate that all required fields are filled (CSV, description, schema fields, API key). Any missing field will show an error message at the top.
   - Once validation passes, processing starts.

7. **Monitoring progress**
   - A **status table** shows each row’s `Row ID`, current **Status** (Queued, Processing, ACCEPT, REJECT, DISCOVERED_NEW, SKIP, ERROR), and a short detail message.
   - A **progress info line** (e.g., “Processed 15 data point(s)…”) updates as rows complete.
   - Streamlit’s spinner indicates that the backend pipeline is running.

8. **Results**
   - When finished:
     - A success message shows how many rows passed final validation and where the **output CSV** was saved.
     - A table of the **final validated rows** is displayed in the UI.
   - If no rows passed the final integrity check, you will see a warning instead.

9. **Metrics**
    - Under **“Metrics Summary”**, you’ll see:
      - Total estimated **cost**
      - Total **input tokens**
      - Total **output tokens**
    - A caption at the bottom shows the path to detailed logs, which detail the reasoning behind each data cleaning choice. (e.g., `..._frontend.log`).

