from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import psycopg
from psycopg.rows import dict_row
import sqlparse

FORBIDDEN_SQL_PATTERNS = [
    r"\bINSERT\b",
    r"\bUPDATE\b",
    r"\bDELETE\b",
    r"\bDROP\b",
    r"\bALTER\b",
    r"\bTRUNCATE\b",
    r"\bGRANT\b",
    r"\bREVOKE\b",
    r"\bCREATE\b",
    r"\bCOMMENT\b",
    r"\bCOPY\b",
    r"\bCALL\b",
    r"\bDO\b",
]


@dataclass(slots=True)
class DatabaseConfig:
    host: str = os.getenv("PGHOST", "localhost")
    port: int = int(os.getenv("PGPORT", "5432"))
    user: str = os.getenv("PGUSER", "postgres")
    password: str = os.getenv("PGPASSWORD", "")
    dbname: str = os.getenv("PGDATABASE", "postgres")
    sslmode: str = os.getenv("PGSSLMODE", "prefer")


class QueryValidationError(ValueError):
    """Raised when the generated SQL query is considered unsafe."""


SCHEMA_QUERY = """
WITH table_metadata AS (
    SELECT
        t.table_schema,
        t.table_name,
        obj_description((quote_ident(t.table_schema) || '.' || quote_ident(t.table_name))::regclass, 'pg_class') AS table_description
    FROM information_schema.tables AS t
    WHERE t.table_type = 'BASE TABLE'
      AND t.table_schema NOT IN ('pg_catalog', 'information_schema')
),
column_metadata AS (
    SELECT
        c.table_schema,
        c.table_name,
        c.column_name,
        c.data_type,
        c.is_nullable,
        c.column_default,
        col_description((quote_ident(c.table_schema) || '.' || quote_ident(c.table_name))::regclass,
                        c.ordinal_position) AS column_description
    FROM information_schema.columns AS c
    WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema')
),
primary_keys AS (
    SELECT
        tc.table_schema,
        tc.table_name,
        string_agg(kcu.column_name, ', ' ORDER BY kcu.ordinal_position) AS primary_key_columns
    FROM information_schema.table_constraints AS tc
    JOIN information_schema.key_column_usage AS kcu
      ON tc.constraint_name = kcu.constraint_name
     AND tc.table_schema = kcu.table_schema
     AND tc.table_name = kcu.table_name
    WHERE tc.constraint_type = 'PRIMARY KEY'
    GROUP BY tc.table_schema, tc.table_name
),
foreign_keys AS (
    SELECT
        tc.table_schema,
        tc.table_name,
        string_agg(
            kcu.column_name || ' -> ' || ccu.table_schema || '.' || ccu.table_name || '.' || ccu.column_name,
            '; ' ORDER BY kcu.ordinal_position
        ) AS foreign_key_details
    FROM information_schema.table_constraints AS tc
    JOIN information_schema.key_column_usage AS kcu
      ON tc.constraint_name = kcu.constraint_name
     AND tc.table_schema = kcu.table_schema
    JOIN information_schema.constraint_column_usage AS ccu
      ON ccu.constraint_name = tc.constraint_name
     AND ccu.table_schema = tc.table_schema
    WHERE tc.constraint_type = 'FOREIGN KEY'
    GROUP BY tc.table_schema, tc.table_name
)
SELECT
    tm.table_schema,
    tm.table_name,
    tm.table_description,
    pk.primary_key_columns,
    fk.foreign_key_details,
    cm.column_name,
    cm.data_type,
    cm.is_nullable,
    cm.column_default,
    cm.column_description
FROM table_metadata AS tm
LEFT JOIN column_metadata AS cm
  ON tm.table_schema = cm.table_schema
 AND tm.table_name = cm.table_name
LEFT JOIN primary_keys AS pk
  ON tm.table_schema = pk.table_schema
 AND tm.table_name = pk.table_name
LEFT JOIN foreign_keys AS fk
  ON tm.table_schema = fk.table_schema
 AND tm.table_name = fk.table_name
ORDER BY tm.table_schema, tm.table_name, cm.column_name;
"""


def build_connection_string(config: DatabaseConfig) -> str:
    return (
        f"host={config.host} port={config.port} dbname={config.dbname} "
        f"user={config.user} password={config.password} sslmode={config.sslmode} "
        "options='-c default_transaction_read_only=on'"
    )


def get_connection(config: DatabaseConfig) -> psycopg.Connection:
    return psycopg.connect(build_connection_string(config), row_factory=dict_row)


def fetch_schema_metadata(config: DatabaseConfig) -> list[dict[str, Any]]:
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_QUERY)
            return list(cur.fetchall())


def format_schema_for_prompt(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No user tables were found in the configured database."

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["table_schema"], row["table_name"])
        if key not in grouped:
            grouped[key] = {
                "table_description": row.get("table_description") or "No description provided.",
                "primary_key_columns": row.get("primary_key_columns") or "None",
                "foreign_key_details": row.get("foreign_key_details") or "None",
                "columns": [],
            }
        if row.get("column_name"):
            grouped[key]["columns"].append(
                {
                    "name": row["column_name"],
                    "data_type": row.get("data_type") or "unknown",
                    "nullable": row.get("is_nullable") or "UNKNOWN",
                    "default": row.get("column_default") or "None",
                    "description": row.get("column_description") or "No description provided.",
                }
            )

    sections: list[str] = []
    for (schema_name, table_name), table_data in grouped.items():
        lines = [
            f"Table: {schema_name}.{table_name}",
            f"Description: {table_data['table_description']}",
            f"Primary key(s): {table_data['primary_key_columns']}",
            f"Foreign key relationship(s): {table_data['foreign_key_details']}",
            "Columns:",
        ]
        for column in table_data["columns"]:
            lines.append(
                "  - "
                f"{column['name']} ({column['data_type']}, nullable={column['nullable']}, default={column['default']}): "
                f"{column['description']}"
            )
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def validate_read_only_sql(query: str) -> str:
    normalized = query.strip().rstrip(";")
    if not normalized:
        raise QueryValidationError("The model returned an empty SQL query.")

    statements = sqlparse.parse(normalized)
    if len(statements) != 1:
        raise QueryValidationError("Only a single SQL statement is allowed.")

    first_token = next(
        (token for token in statements[0].flatten() if not token.is_whitespace),
        None,
    )
    if first_token is None:
        raise QueryValidationError("The SQL query could not be parsed.")

    if first_token.value.upper() not in {"SELECT", "WITH"}:
        raise QueryValidationError("Only SELECT queries and CTEs that end in SELECT are allowed.")

    upper_query = normalized.upper()
    for pattern in FORBIDDEN_SQL_PATTERNS:
        if re.search(pattern, upper_query):
            raise QueryValidationError("The generated SQL contains a forbidden write or DDL operation.")

    return normalized


def execute_read_only_query(config: DatabaseConfig, query: str) -> list[dict[str, Any]]:
    safe_query = validate_read_only_sql(query)
    with get_connection(config) as conn:
        with conn.cursor() as cur:
            cur.execute(safe_query)
            return list(cur.fetchall())
