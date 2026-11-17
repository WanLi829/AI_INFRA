"""Record read-only BayesDL observations supplied by the scheduled browser check."""

import argparse
import json
import os
import sqlite3
from pathlib import Path


def record(db, key, value):
    db.execute("CREATE TABLE IF NOT EXISTS observations (name TEXT PRIMARY KEY, value TEXT)")
    previous = db.execute("SELECT value FROM observations WHERE name = ?", (key,)).fetchone()
    db.execute("INSERT OR REPLACE INTO observations VALUES (?, ?)", (key, value))
    return previous is None or previous[0] != value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", choices=["ready", "busy", "missing", "unknown"], default="unknown")
    parser.add_argument("--stock", type=int, help="Confirmed count of available dedicated 4090D 24GB cards")
    parser.add_argument("--login-required", action="store_true")
    parser.add_argument("--db", type=Path, default=Path(os.getenv("LOCALAPPDATA") or Path.home())
                        / "BayesDL-Watch" / "4090d.sqlite3")
    args = parser.parse_args()
    if args.stock is not None and args.stock < 0:
        parser.error("stock must be nonnegative")
    args.db.parent.mkdir(parents=True, exist_ok=True)
    events = []
    with sqlite3.connect(args.db, timeout=15) as db:
        db.execute("BEGIN IMMEDIATE")
        if args.login_required:
            if record(db, "login", "required"):
                events.append("login_required")
        else:
            if args.instance != "unknown" or args.stock is not None:
                record(db, "login", "ok")
            if args.instance != "unknown":
                if record(db, "instance", args.instance) and args.instance == "ready":
                    events.append("instance_may_start")
            if args.stock is not None:
                available = args.stock > 0
                if record(db, "stock_available", str(available)) and available:
                    events.append("new_stock")
    print(json.dumps(events))


if __name__ == "__main__":
    main()
