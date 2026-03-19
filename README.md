# Phi-3 PostgreSQL Natural Language Query App

This repository contains a Streamlit application that turns natural-language questions into safe PostgreSQL `SELECT` queries by using a locally running Phi-3 model served through Ollama.

## Features

- Streamlit user interface for asking plain-English questions.
- Ollama integration for SQL generation and result summarization.
- PostgreSQL schema extraction so the model receives table, column, key, and relationship context.
- Read-only SQL validation to block destructive or multi-statement queries.
- Resilient model-response parsing that can recover SQL from markdown fences or non-JSON fallback output.
- Query history and optional SQL transparency for end users.

## Getting Started

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Make sure PostgreSQL is reachable locally and the configured user has read-only access.

3. Make sure Ollama is running and the desired Phi-3 model is installed locally, for example:

   ```bash
   ollama pull phi3:mini
   ```

4. Set environment variables if you do not want to type the database settings into the sidebar every time:

   ```bash
   export PGHOST=localhost
   export PGPORT=5432
   export PGUSER=postgres
   export PGPASSWORD=postgres
   export PGDATABASE=postgres
   export PGSSLMODE=prefer
   export OLLAMA_MODEL=phi3:mini
   ```

5. Launch the app:

   ```bash
   streamlit run app.py
   ```

## How It Works

1. The app connects to PostgreSQL and introspects user schemas, columns, descriptions, primary keys, and foreign keys.
2. The schema metadata is formatted into a prompt and sent to the local Phi-3 model using Ollama.
3. The model is instructed to return JSON, but the app can also recover SQL from markdown-fenced or plain SQL fallback responses.
4. The app validates that the SQL is a single read-only statement before execution.
5. The SQL runs against PostgreSQL in a read-only transaction.
6. The result rows are shown in a table and summarized by the model for non-technical users.

## Safety Notes

- The application only allows `SELECT` statements (including `WITH` CTEs that resolve to `SELECT`).
- Destructive SQL keywords such as `DELETE`, `DROP`, `ALTER`, or `INSERT` are rejected before execution.
- PostgreSQL connections are opened with `default_transaction_read_only=on` for defense in depth.

## Extensibility

The code is split so you can replace or extend the local model integration later:

- `app.py` contains the Streamlit UI and workflow orchestration.
- `db_utils.py` contains PostgreSQL connectivity, schema discovery, SQL validation, and execution helpers.
- `llm_utils.py` contains Ollama prompts and JSON response handling.
