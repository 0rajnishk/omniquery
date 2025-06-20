#!/usr/bin/env python3
"""
Convert every sheet in an Excel workbook to tables inside a SQLite database.

Usage
-----
python excel_to_sqlite.py workbook.xlsx           # creates workbook.db
python excel_to_sqlite.py workbook.xlsx --db out.db
"""

import argparse
import os
import re
import sqlite3
import sys

import pandas as pd


def sanitize(name: str) -> str:
    """
    Make a string safe for use as a SQL table name:
      • trim spaces
      • lower-case
      • spaces → underscores
      • drop characters that aren’t alphanumeric or “_”
    """
    name = name.strip().lower()
    name = re.sub(r"\s+", "_", name)
    return re.sub(r"[^\w]", "", name)

def sanitize_column(col: str) -> str:
    """
    Sanitize a column name:
      - Replace spaces with underscores
      - If there is a capital letter followed by a small letter, insert underscore before capital
      - Convert to lower-case
      - Remove non-alphanumeric/underscore
    """
    # Replace spaces with underscores
    col = re.sub(r"\s+", "_", col)
    # Insert underscore before capital letters that are followed by lowercase (for CamelCase)
    col = re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', col)
    # Convert to lower-case
    col = col.lower()
    # Remove non-alphanumeric/underscore
    col = re.sub(r"[^\w]", "", col)
    return col

def excel_to_sqlite(xls_path: str, db_path: str | None = None) -> str:
    """Read *xls_path* and write an SQLite file.  
    Returns the path to the database created."""
    if not os.path.isfile(xls_path):
        sys.exit(f"[ERROR] Excel file not found: {xls_path}")

    if db_path is None:
        db_path = f"{os.path.splitext(os.path.basename(xls_path))[0]}.db"

    # Open workbook once, then iterate over all its sheets
    excel = pd.ExcelFile(xls_path)
    with sqlite3.connect(db_path) as conn:
        for sheet in excel.sheet_names:
            df = excel.parse(sheet)
            table_name = sanitize(sheet)

            # Sanitize column names as per requirements
            df.columns = [sanitize_column(col) for col in df.columns]

            # Write DataFrame → SQL table (replace if it already exists)
            df.to_sql(table_name, conn, if_exists="replace", index=False)

            print(f"  → Sheet '{sheet}' saved as table '{table_name}'")

    print(f"\nDone! SQLite database created at: {os.path.abspath(db_path)}")
    return db_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an Excel workbook into a SQLite database."
    )
    parser.add_argument("excel", help="Path to the .xlsx or .xls file")
    parser.add_argument(
        "--db",
        help="Output .db file name (default: <excel-filename>.db in the same folder)",
    )
    args = parser.parse_args()

    excel_to_sqlite(args.excel, args.db)


if __name__ == "__main__":
    main()
