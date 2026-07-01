"""db/seed.py — (re)build the customers table from the model + test split.

Usage (from the project root):
    python -m db.seed            # seed only if empty
    python -m db.seed --force    # wipe and rebuild
"""

import argparse

from db.connection import init_db
from db.customers import seed_customers


def main(force: bool) -> None:
    init_db()
    n = seed_customers(force=force)
    print(f"customers table ready — {n} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed the customers table.")
    parser.add_argument("--force", action="store_true", help="wipe and rebuild")
    args = parser.parse_args()
    main(args.force)
