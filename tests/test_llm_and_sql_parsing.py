from __future__ import annotations

import unittest

from db_utils import QueryValidationError, validate_read_only_sql
from llm_utils import parse_sql_generation_response, parse_summary_response


class SQLAndLLMParsingTests(unittest.TestCase):
    def test_recovers_sql_from_markdown_fence(self) -> None:
        raw_response = """```sql
SELECT employee_id
FROM admin.employees
ORDER BY salary::numeric DESC
LIMIT 1;
```"""

        parsed = parse_sql_generation_response(raw_response)

        self.assertEqual(
            parsed["sql"],
            "SELECT employee_id\nFROM admin.employees\nORDER BY salary::numeric DESC\nLIMIT 1",
        )
        self.assertIn("Recovered SQL", parsed["rationale"])

    def test_parses_json_wrapped_in_markdown(self) -> None:
        raw_response = """```json
{"sql": "SELECT employee_id FROM admin.employees ORDER BY salary::numeric DESC LIMIT 1;", "rationale": "Highest salary first."}
```"""

        parsed = parse_sql_generation_response(raw_response)

        self.assertEqual(
            parsed["sql"],
            "SELECT employee_id FROM admin.employees ORDER BY salary::numeric DESC LIMIT 1;",
        )
        self.assertEqual(parsed["rationale"], "Highest salary first.")

    def test_summary_falls_back_to_plain_text(self) -> None:
        parsed = parse_summary_response("Top salary belongs to employee 42.")

        self.assertEqual(parsed["summary"], "Top salary belongs to employee 42.")
        self.assertEqual(parsed["highlights"], [])

    def test_sql_validation_normalizes_fenced_queries(self) -> None:
        validated = validate_read_only_sql("```sql\nSELECT employee_id FROM admin.employees;\n```")

        self.assertEqual(validated, "SELECT employee_id FROM admin.employees")

    def test_sql_validation_rejects_write_queries(self) -> None:
        with self.assertRaises(QueryValidationError):
            validate_read_only_sql("DELETE FROM admin.employees")


if __name__ == "__main__":
    unittest.main()
