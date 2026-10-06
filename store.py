"""SQLite store: visits (for people counting) and security events."""
import datetime
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS visits(
    id INTEGER PRIMARY KEY,
    track_id INTEGER,
    entered_at REAL,          -- video seconds
    exited_at REAL,           -- video seconds (NULL if still inside)
    wall_entered TEXT,
    wall_exited TEXT
);
CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY,
    type TEXT,                -- 'loitering' | 'weapon'
    track_id INTEGER,
    video_time REAL,
    wall_time TEXT,
    details TEXT,
    snapshot TEXT
);
CREATE INDEX IF NOT EXISTS idx_visits_entered ON visits(wall_entered);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(type);
"""


def init_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")


def log_entry(conn, track_id, t_video):
    cur = conn.execute(
        "INSERT INTO visits(track_id, entered_at, wall_entered) VALUES (?,?,?)",
        (track_id, t_video, _now()))
    conn.commit()
    return cur.lastrowid


def log_exit(conn, visit_id, t_video):
    conn.execute(
        "UPDATE visits SET exited_at=?, wall_exited=? WHERE id=?",
        (t_video, _now(), visit_id))
    conn.commit()


def log_event(conn, etype, track_id, t_video, details="", snapshot=None):
    conn.execute(
        "INSERT INTO events(type, track_id, video_time, wall_time, details, snapshot)"
        " VALUES (?,?,?,?,?,?)",
        (etype, track_id, t_video, _now(), details, snapshot))
    conn.commit()


def weekly_counts(conn, n=12):
    """[(week_label, entries)] for the last n ISO weeks, oldest first."""
    rows = conn.execute("""
        SELECT strftime('%Y-W%W', wall_entered) AS wk, COUNT(*)
        FROM visits GROUP BY wk ORDER BY wk DESC LIMIT ?""", (n,)).fetchall()
    return list(reversed(rows))


def monthly_counts(conn, n=12):
    """[(month_label, entries)] for the last n months, oldest first."""
    rows = conn.execute("""
        SELECT strftime('%Y-%m', wall_entered) AS mo, COUNT(*)
        FROM visits GROUP BY mo ORDER BY mo DESC LIMIT ?""", (n,)).fetchall()
    return list(reversed(rows))


def totals(conn):
    v = conn.execute("SELECT COUNT(*), COUNT(exited_at) FROM visits").fetchone()
    e = conn.execute(
        "SELECT type, COUNT(*) FROM events GROUP BY type").fetchall()
    return {"visits": v[0], "exits": v[1], "events": dict(e)}
