"""
Load the raw synthetic CSVs into DuckDB as source tables.

This stands in for a warehouse ingestion step (e.g. Snowflake COPY INTO / an
Airbyte or Fivetran sync landing raw tables). Everything is loaded as-is,
with no cleaning or type coercion -- that happens in the dbt staging layer
so the transformation logic lives in one place (see dbt/transaction_analytics).

Run:
    python python/etl/load_to_duckdb.py
"""

from __future__ import annotations

import duckdb

DB_PATH = "warehouse/transaction_analytics.duckdb"
RAW_SCHEMA = "raw"

TABLES = {
    "customers": "data/customers.csv",
    "transactions": "data/transactions.csv",
    "support_tickets": "data/support_tickets.csv",
}


def main() -> None:
    con = duckdb.connect(DB_PATH)
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA}")

    for table, csv_path in TABLES.items():
        con.execute(
            f"""
            CREATE OR REPLACE TABLE {RAW_SCHEMA}.{table} AS
            SELECT * FROM read_csv_auto('{csv_path}', header=True)
            """
        )
        n_rows = con.execute(f"SELECT COUNT(*) FROM {RAW_SCHEMA}.{table}").fetchone()[0]
        print(f"  loaded {n_rows:,} rows -> {RAW_SCHEMA}.{table}")

    con.close()
    print(f"\nDone. Warehouse file: {DB_PATH}")


if __name__ == "__main__":
    main()
