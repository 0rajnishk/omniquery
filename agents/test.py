import sqlite3
import os
from typing import List

DB_PATH = "../data/HospitalPatientRecordsDataset.db"
assert os.path.exists(DB_PATH), f"SQLite database not found at {DB_PATH}"


def _get_schema_info() -> str:
    """Return schema (table, columns, sample categorical values)."""
    info_lines: List[str] = []

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # List user-defined tables
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        )
        tables = [row["name"] for row in cursor.fetchall()]

        for table in tables:
            info_lines.append(f"\nTable: {table}")
            cursor.execute(f"PRAGMA table_info('{table}')")
            columns = cursor.fetchall()

            for col in columns:
                col_name = col["name"]
                col_type = col["type"] or "UNKNOWN"
                samples = ""

                # For text-like columns, fetch up to 10 distinct sample values
                if col_type.upper() in ("TEXT", ""):
                    try:
                        cursor.execute(
                            f"SELECT DISTINCT {col_name} FROM {table} WHERE {col_name} IS NOT NULL LIMIT 10"
                        )
                        vals = [str(r[0]) for r in cursor.fetchall()]
                        if vals:
                            samples = f" (sample values: {', '.join(vals)})"
                    except sqlite3.OperationalError:
                        samples = " (error fetching sample values)"

                info_lines.append(f"  - {col_name} ({col_type}){samples}")

    return "\n".join(info_lines)


if __name__ == "__main__":
    print(_get_schema_info())
