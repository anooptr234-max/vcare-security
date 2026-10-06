#!/usr/bin/env python3
"""Visitor analytics: weekly / monthly customer counts from the security db.

Usage:
    python report.py --db security.db --period week
    python report.py --db security.db --period month --last 6
"""
import argparse
import sqlite3

from store import weekly_counts, monthly_counts, totals


def main():
    ap = argparse.ArgumentParser(description="VCARE visitor analytics")
    ap.add_argument("--db", default="security.db")
    ap.add_argument("--period", choices=["week", "month"], default="week")
    ap.add_argument("--last", type=int, default=12)
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    rows = weekly_counts(conn, args.last) if args.period == "week" \
        else monthly_counts(conn, args.last)
    label = "ISO week" if args.period == "week" else "Month"
    print(f"\nCustomer visits per {args.period}  ({args.db})")
    print(f"{label:>12} | {'visits':>6}")
    print("-" * 23)
    total = 0
    for lab, n in rows:
        print(f"{lab:>12} | {n:>6}")
        total += n
    print("-" * 23)
    print(f"{'TOTAL':>12} | {total:>6}")
    t = totals(conn)
    print(f"\nAll-time visits: {t['visits']}  exits: {t['exits']}  "
          f"security events: {t['events'] or 'none'}")
    conn.close()


if __name__ == "__main__":
    main()
