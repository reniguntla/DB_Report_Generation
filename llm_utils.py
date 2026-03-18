from __future__ import annotations

import json
from typing import Any

import ollama

SQL_SYSTEM_PROMPT = """You are an expert PostgreSQL analyst.
Translate user requests into a single safe PostgreSQL SELECT statement.
Rules:
- Only generate read-only SQL.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, COPY, GRANT, REVOKE, CALL, or DO.
- Use only the tables and columns present in the provided schema.
- Prefer explicit JOIN conditions using foreign-key relationships.
- Keep the query efficient and easy to read.
- Return valid JSON with keys: sql, rationale.
"""

RESULT_SYSTEM_PROMPT = """You explain PostgreSQL query results to non-technical users.
Keep the explanation concise, note key observations, and mention if the result set is empty.
Return valid JSON with keys: summary, highlights.
"""


class LLMResponseError(ValueError):
    """Raised when the local model returns malformed output."""


def _chat(model: str, system_prompt: str, user_prompt: str) -> str:
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1},
    )
    return response["message"]["content"]


def _load_json_response(raw_response: str) -> dict[str, Any]:
    try:
        return json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(
            "The local model returned a response that was not valid JSON. "
            f"Raw output: {raw_response}"
        ) from exc


def generate_sql(model: str, schema_context: str, user_question: str) -> dict[str, str]:
    prompt = f"""
Database schema:
{schema_context}

User question:
{user_question}

Return JSON only.
""".strip()
    payload = _load_json_response(_chat(model, SQL_SYSTEM_PROMPT, prompt))
    sql = str(payload.get("sql", "")).strip()
    rationale = str(payload.get("rationale", "")).strip()
    if not sql:
        raise LLMResponseError("The local model did not return an SQL query.")
    return {"sql": sql, "rationale": rationale}


def summarize_results(
    model: str,
    user_question: str,
    sql_query: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    serialized_rows = json.dumps(rows[:20], default=str, indent=2)
    row_count = len(rows)
    prompt = f"""
User question:
{user_question}

Executed SQL:
{sql_query}

Row count: {row_count}
Sample rows (up to 20):
{serialized_rows}

Return JSON only.
""".strip()
    payload = _load_json_response(_chat(model, RESULT_SYSTEM_PROMPT, prompt))
    return {
        "summary": str(payload.get("summary", "")).strip(),
        "highlights": payload.get("highlights", []),
    }
