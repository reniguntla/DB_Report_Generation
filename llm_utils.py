from __future__ import annotations

import importlib
import importlib.util
import json
import re
from typing import Any

_OLLAMA_SPEC = importlib.util.find_spec("ollama")
ollama = importlib.import_module("ollama") if _OLLAMA_SPEC else None

SQL_SYSTEM_PROMPT = """You are an expert PostgreSQL analyst.
Translate user requests into a single safe PostgreSQL SELECT statement.
Rules:
- Only generate read-only SQL.
- Never use INSERT, UPDATE, DELETE, DROP, ALTER, CREATE, TRUNCATE, COPY, GRANT, REVOKE, CALL, or DO.
- Use only the tables and columns present in the provided schema.
- Prefer explicit JOIN conditions using foreign-key relationships.
- Do not group by or partition by extra columns unless the user asked for that breakdown.
- Return the minimal result needed to answer the question.
- Respond with raw JSON only and never wrap the answer in markdown code fences.
- Return valid JSON with keys: sql, rationale.
"""

RESULT_SYSTEM_PROMPT = """You explain PostgreSQL query results to non-technical users.
Keep the explanation concise, note key observations, and mention if the result set is empty.
Respond with raw JSON only and never wrap the answer in markdown code fences.
Return valid JSON with keys: summary, highlights.
"""

SQL_KEYWORDS = (
    "SELECT",
    "WITH",
    "FROM",
    "WHERE",
    "GROUP BY",
    "ORDER BY",
    "LIMIT",
    "JOIN",
)


class LLMResponseError(ValueError):
    """Raised when the local model returns malformed output."""


class LLMDependencyError(RuntimeError):
    """Raised when Ollama support is unavailable in the local Python environment."""


def _chat(model: str, system_prompt: str, user_prompt: str) -> str:
    if ollama is None:
        raise LLMDependencyError(
            "The 'ollama' Python package is not installed. Install dependencies from requirements.txt before running the app."
        )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0.1},
    )
    return response["message"]["content"]


def _strip_code_fences(text: str) -> str:
    fenced_match = re.search(r"```(?:json|sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced_match:
        return fenced_match.group(1).strip()
    return text.strip()


def _extract_json_object(text: str) -> str | None:
    stripped = _strip_code_fences(text)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            _, end_index = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        return stripped[index : index + end_index]
    return None


def _looks_like_sql(text: str) -> bool:
    upper_text = text.upper()
    return any(keyword in upper_text for keyword in SQL_KEYWORDS)


def _extract_sql_candidate(text: str) -> str | None:
    stripped = _strip_code_fences(text)
    if _looks_like_sql(stripped):
        return stripped.rstrip(";")

    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    sql_lines = [line for line in lines if _looks_like_sql(line)]
    if sql_lines:
        return " ".join(sql_lines).rstrip(";")
    return None


def _normalize_highlights(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _load_json_response(raw_response: str) -> dict[str, Any]:
    json_payload = _extract_json_object(raw_response)
    if json_payload:
        try:
            return json.loads(json_payload)
        except json.JSONDecodeError:
            pass

    raise LLMResponseError(
        "The local model returned a response that was not valid JSON. "
        f"Raw output: {raw_response}"
    )


def parse_sql_generation_response(raw_response: str) -> dict[str, str]:
    try:
        payload = _load_json_response(raw_response)
        sql = str(payload.get("sql", "")).strip()
        rationale = str(payload.get("rationale", "")).strip()
        if sql:
            return {"sql": sql, "rationale": rationale}
    except LLMResponseError:
        sql_candidate = _extract_sql_candidate(raw_response)
        if sql_candidate:
            return {
                "sql": sql_candidate,
                "rationale": "Recovered SQL from a non-JSON model response.",
            }
        raise

    raise LLMResponseError("The local model did not return an SQL query.")


def parse_summary_response(raw_response: str) -> dict[str, Any]:
    try:
        payload = _load_json_response(raw_response)
        return {
            "summary": str(payload.get("summary", "")).strip(),
            "highlights": _normalize_highlights(payload.get("highlights", [])),
        }
    except LLMResponseError:
        fallback_summary = _strip_code_fences(raw_response)
        if fallback_summary:
            return {"summary": fallback_summary, "highlights": []}
        raise


def generate_sql(model: str, schema_context: str, user_question: str) -> dict[str, str]:
    prompt = f"""
Database schema:
{schema_context}

User question:
{user_question}

Return JSON only.
""".strip()
    return parse_sql_generation_response(_chat(model, SQL_SYSTEM_PROMPT, prompt))


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
    return parse_summary_response(_chat(model, RESULT_SYSTEM_PROMPT, prompt))
