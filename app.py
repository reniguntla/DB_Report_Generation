from __future__ import annotations

import os
from dataclasses import asdict

import pandas as pd
import streamlit as st

from db_utils import (
    DatabaseConfig,
    DatabaseDependencyError,
    QueryValidationError,
    execute_read_only_query,
    fetch_schema_metadata,
    format_schema_for_prompt,
    validate_read_only_sql,
)
from llm_utils import LLMDependencyError, LLMResponseError, generate_sql, summarize_results

st.set_page_config(page_title="Phi-3 PostgreSQL Query Assistant", page_icon="🗄️", layout="wide")

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
EXAMPLE_QUESTIONS = [
    "List the top 5 employees with the highest salary including employee ID, name, and salary.",
    "Show total sales by month for the current year.",
    "Which customers have placed more than 10 orders?",
]


@st.cache_data(ttl=300, show_spinner=False)
def load_schema(config_dict: dict[str, str]) -> tuple[list[dict], str]:
    config = DatabaseConfig(**config_dict)
    rows = fetch_schema_metadata(config)
    return rows, format_schema_for_prompt(rows)


def initialize_session_state() -> None:
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("user_question", EXAMPLE_QUESTIONS[0])


initialize_session_state()

st.title("🗄️ Phi-3 Natural Language PostgreSQL Explorer")
st.caption(
    "Ask questions in plain English, generate safe SQL with a local Ollama-hosted Phi-3 model, and inspect the results."
)

with st.sidebar:
    st.header("Configuration")
    model_name = st.text_input("Ollama model tag", value=DEFAULT_MODEL, help="Use the exact model tag available in your local Ollama instance.")

    with st.form("db_config_form"):
        host = st.text_input("Host", value=DatabaseConfig.host)
        port = st.number_input("Port", value=DatabaseConfig.port, min_value=1, max_value=65535)
        user = st.text_input("User", value=DatabaseConfig.user)
        password = st.text_input("Password", value=DatabaseConfig.password, type="password")
        dbname = st.text_input("Database", value=DatabaseConfig.dbname)
        sslmode = st.selectbox("SSL mode", options=["disable", "allow", "prefer", "require", "verify-ca", "verify-full"], index=2)
        connect_clicked = st.form_submit_button("Refresh schema")

    config = DatabaseConfig(
        host=host,
        port=int(port),
        user=user,
        password=password,
        dbname=dbname,
        sslmode=sslmode,
    )

    st.subheader("Example questions")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, use_container_width=True):
            st.session_state["user_question"] = example

if connect_clicked or "schema_context" not in st.session_state:
    try:
        schema_rows, schema_context = load_schema(asdict(config))
        st.session_state["schema_rows"] = schema_rows
        st.session_state["schema_context"] = schema_context
        st.sidebar.success("Schema loaded successfully.")
    except Exception as exc:  # noqa: BLE001
        st.session_state.pop("schema_rows", None)
        st.session_state.pop("schema_context", None)
        st.sidebar.error(f"Unable to load schema metadata: {exc}")

question = st.text_area(
    "Ask a question about your database",
    key="user_question",
    height=120,
    placeholder="Example: List the top 5 employees with the highest salary including employee ID, name, and salary.",
)

show_sql = st.checkbox("Show generated SQL", value=True)
submitted = st.button("Run query", type="primary", use_container_width=True)

schema_tab, results_tab, history_tab = st.tabs(["Schema", "Results", "History"])

with schema_tab:
    if st.session_state.get("schema_context"):
        st.text_area("Schema context sent to the model", st.session_state["schema_context"], height=400)
    else:
        st.info("Load the schema from the sidebar to give the model database context.")

with results_tab:
    if not submitted:
        st.info("Enter a question and click **Run query** to generate SQL and view results.")
    else:
        if not question.strip():
            st.warning("Please enter a natural-language question.")
        elif not st.session_state.get("schema_context"):
            st.error("Schema metadata is not loaded yet. Refresh the schema from the sidebar first.")
        else:
            try:
                with st.spinner("Generating SQL with Phi-3 via Ollama..."):
                    generation = generate_sql(model_name, st.session_state["schema_context"], question)

                safe_sql = validate_read_only_sql(generation["sql"])

                with st.spinner("Executing query against PostgreSQL..."):
                    rows = execute_read_only_query(config, safe_sql)
                    dataframe = pd.DataFrame(rows)

                with st.spinner("Summarizing results..."):
                    summary = summarize_results(model_name, question, safe_sql, rows)

                if show_sql:
                    st.subheader("Generated SQL")
                    st.code(safe_sql, language="sql")

                if generation["rationale"]:
                    st.subheader("Why this SQL was generated")
                    st.write(generation["rationale"])

                st.subheader("Query results")
                if dataframe.empty:
                    st.info("The query ran successfully but returned no rows.")
                else:
                    st.dataframe(dataframe, use_container_width=True)
                    st.caption(f"Returned {len(dataframe)} row(s).")

                st.subheader("AI explanation")
                if summary["summary"]:
                    st.write(summary["summary"])
                highlights = summary.get("highlights") or []
                if isinstance(highlights, list) and highlights:
                    st.markdown("**Highlights**")
                    for item in highlights:
                        st.markdown(f"- {item}")

                st.session_state["history"].insert(
                    0,
                    {
                        "question": question,
                        "sql": safe_sql,
                        "row_count": len(rows),
                        "summary": summary.get("summary", ""),
                    },
                )
            except (DatabaseDependencyError, LLMDependencyError, QueryValidationError, LLMResponseError) as exc:
                st.error(f"Safety or model output validation failed: {exc}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Query execution failed: {exc}")

with history_tab:
    history = st.session_state.get("history", [])
    if not history:
        st.info("Your query history will appear here after you run a question.")
    else:
        for item in history[:10]:
            with st.expander(item["question"]):
                st.code(item["sql"], language="sql")
                st.write(f"Rows returned: {item['row_count']}")
                if item["summary"]:
                    st.write(item["summary"])
