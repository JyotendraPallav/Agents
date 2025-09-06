# TeamPulse.py
# =============================================================================
# README
# =============================================================================
# TeamPulse — Lightweight Team Kanban & Utilization Tracker
#
# Run:
#   Create your .env file
#   python TeamPulse.py
#
# Requirements:
#   - Python 3.9+
#   - pip install gradio pandas matplotlib
#
# Features:
#   - CSV-backed persistence (./data/*.csv), auto-seeded on first run
#   - Data model:
#       members(member_id, name, role, weekly_capacity_hours, is_deleted)
#       tasks(task_id, title, description, assignee_id, status, priority,
#             estimate_hours, created_at, due_date, tags, dependencies, is_deleted)
#       time_logs(log_id, task_id, member_id, date, hours_spent, note, is_deleted)
#   - CRUD for Tasks, Time Logs, Members (soft delete)
#   - Kanban board with filters (assignee, priority, tags, date range, search)
#   - Time tracking: add logs + recent logs view + per-task hours vs estimate
#   - Reports:
#       * Utilization by member (date range, capacity prorated by days/7)
#       * Heatmap (member × ISO week) of hours
#       * Burndown (remaining hours for tasks due in window)
#       * Throughput (tasks Done per week)
#       * Cycle time (Created→Done) with optional “ignore weekends”
#       * WIP count by day
#       * Quick pivots by status/priority/tag/assignee
#   - Export current Tasks / Time Logs / Metrics to CSV
#   - Download charts as PNG
#
# Notes:
#   - Charts use matplotlib only, one figure per chart, default styling/colors.
#   - Drag-and-drop is not required. Status can be changed via quick edit in Kanban.
#   - If anything’s ambiguous, sensible defaults are used for a small team UX.
# =============================================================================

from __future__ import annotations
import os
import io
import math
import json
import shutil
from datetime import datetime, date, timedelta
from typing import Tuple, List, Optional, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

import glob

from dotenv import load_dotenv
load_dotenv()   # this loads variables from .env into os.environ

# --------------------------
# Constants & file locations
# --------------------------
APP_NAME = "TeamPulse"
DATA_DIR = "./data"
EXPORT_DIR = os.path.join(DATA_DIR, "exports")
MEMBERS_CSV = os.path.join(DATA_DIR, "members.csv")
TASKS_CSV = os.path.join(DATA_DIR, "tasks.csv")
LOGS_CSV = os.path.join(DATA_DIR, "time_logs.csv")

STATUSES = ["Backlog", "In-Progress", "Review", "Done"]
PRIORITIES = ["P1", "P2", "P3", "P4"]

# --- ADD: for simple multi-user sharing via a CSV inbox ---

# Optional: let teams override data root via env if you later want a shared data dir
# (keep your existing DATA_DIR; this only adds an inbox alongside it)
INBOX_DIR = os.path.expandvars(os.path.expanduser(
    os.getenv("KANBAN_INBOX_DIR", os.path.join(DATA_DIR, "_inbox"))
))
PROCESSED_DIR = os.path.join(INBOX_DIR, "processed")




# ---------------
# Utility helpers
# ---------------
def _ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(EXPORT_DIR, exist_ok=True)
    os.makedirs(INBOX_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)


def _today_str() -> str:
    return date.today().isoformat()

def _parse_date_safe(s: str | date | pd.Timestamp) -> pd.Timestamp:
    """Robustly parse date-like input into pandas.Timestamp (date-normalized)."""
    if pd.isna(s) or s is None or s == "":
        return pd.NaT
    if isinstance(s, (pd.Timestamp, )):
        return pd.Timestamp(s).normalize()
    if isinstance(s, date):
        return pd.Timestamp(s).normalize()
    return pd.to_datetime(str(s), errors="coerce").normalize()

def _date_range_days(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Inclusive number of days between two timestamps. Returns 0 if invalid."""
    if pd.isna(start) or pd.isna(end):
        return 0
    if end < start:
        return 0
    return int((end - start).days) + 1

def _biz_days_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Business days inclusive between two dates (Mon–Fri)."""
    if pd.isna(start) or pd.isna(end) or end < start:
        return 0
    # pandas.bdate_range is inclusive of start/end by default (closed='both')
    return len(pd.bdate_range(start, end))

def _next_id(df: pd.DataFrame, col: str) -> int:
    if df.empty or col not in df.columns or df[col].dropna().empty:
        return 1
    try:
        return int(df[col].astype(int).max()) + 1
    except Exception:
        # fallback for non-numeric stray values
        numeric = pd.to_numeric(df[col], errors="coerce")
        return int(numeric.max(skipna=True) or 0) + 1

def _split_csv_ids(s: str) -> List[int]:
    if not s or pd.isna(s):
        return []
    out = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            # ignore non-integer tokens
            pass
    return out

def _join_ids(ids: List[int]) -> str:
    return ",".join(str(i) for i in ids) if ids else ""

def _normalize_bool_col(df: pd.DataFrame, col: str):
    if col not in df.columns:
        df[col] = False
    df[col] = df[col].fillna(False).astype(bool)
    return df

def _safe_float(x, default=0.0) -> float:
    try:
        f = float(x)
        if math.isnan(f):
            return default
        return f
    except Exception:
        return default

def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _task_label_choices(df: pd.DataFrame | None = None):
    """Labels like '123 — Build Kanban UI' for add-log dropdowns."""
    t = load_tasks() if df is None else df
    t = t[~t["is_deleted"]]
    return [f"{r.task_id} — {r.title}" for _, r in t.iterrows()]

def _task_id_choices(df: pd.DataFrame | None = None):
    """Plain task_id list for edit forms."""
    t = load_tasks() if df is None else df
    t = t[~t["is_deleted"]]
    return [int(r.task_id) for _, r in t.iterrows()]


#--------------------------------------------------------------------
# ---------- SHARED INBOX HELPERS (federated time-log sharing) ----------
#--------------------------------------------------------------------------

def _log_fingerprint(row: pd.Series) -> tuple:
    """
    A stable signature for deduping logs across users.
    (task_id, member_id, date, hours_spent, note)
    """
    return (
        int(row["task_id"]),
        int(row["member_id"]),
        str(row["date"]),
        float(row["hours_spent"]),
        str(row.get("note", "")),
    )

def push_my_logs_to_inbox(user_label: str) -> str:
    """
    Export ALL non-deleted logs from this machine into the shared inbox.
    Duplicates are harmless—merge will dedupe by fingerprint.
    """
    os.makedirs(INBOX_DIR, exist_ok=True)
    if not user_label or not str(user_label).strip():
        return "❌ Please provide your name/initials before pushing."

    logs = load_logs()
    if logs.empty:
        return "ℹ️ No logs to push."

    outgoing = logs[~logs["is_deleted"]].copy()
    # Ensure required columns exist
    for col in EXPECTED_LOGS_COLS:
        if col not in outgoing.columns:
            outgoing[col] = np.nan

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"time_logs_{user_label.replace(' ','_')}_{ts}.csv"
    path = os.path.join(INBOX_DIR, fname)
    outgoing.to_csv(path, index=False)
    return f"✅ Pushed {len(outgoing)} logs to inbox: {path}"

def pull_and_merge_inbox_logs() -> tuple[str, pd.DataFrame]:
    """
    Read all 'time_logs_*.csv' from the shared inbox, append only NEW logs
    (deduped by fingerprint against existing), assign fresh log_id, and
    move processed files to /processed.
    Returns (message, recent_logs_df)
    """
    os.makedirs(INBOX_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    logs = load_logs()
    existing_fps = set()
    if not logs.empty:
        existing_fps = { _log_fingerprint(r) for _, r in logs.iterrows() }

    new_rows = []
    files = sorted(glob.glob(os.path.join(INBOX_DIR, "time_logs_*.csv")))
    processed_count = 0
    imported_count = 0

    for fpath in files:
        try:
            df = pd.read_csv(fpath)
        except Exception:
            # Skip unreadable file
            continue
        if df.empty:
            shutil.move(fpath, os.path.join(PROCESSED_DIR, os.path.basename(fpath)))
            processed_count += 1
            continue

        # Normalize minimal schema
        for col in EXPECTED_LOGS_COLS:
            if col not in df.columns:
                df[col] = np.nan

        # Coerce types/alignment with our loader
        df["task_id"] = pd.to_numeric(df["task_id"], errors="coerce").fillna(0).astype(int)
        df["member_id"] = pd.to_numeric(df["member_id"], errors="coerce").fillna(0).astype(int)
        df["date"] = df["date"].apply(_parse_date_safe).dt.date.astype(str)
        df["hours_spent"] = pd.to_numeric(df["hours_spent"], errors="coerce").fillna(0.0).astype(float)
        df["note"] = df["note"].fillna("").astype(str)
        df = _normalize_bool_col(df, "is_deleted")
        df = df[~df["is_deleted"]]

        # Only accept rows that also pass our validator against current tasks/members
        tasks = load_tasks()
        members = load_members()

        for _, r in df.iterrows():
            # skip invalid rows
            ok, _ = validate_time_log(
                {
                    "log_id": 0,
                    "task_id": int(r["task_id"]),
                    "member_id": int(r["member_id"]),
                    "date": r["date"],
                    "hours_spent": float(r["hours_spent"]),
                    "note": r.get("note", ""),
                    "is_deleted": False,
                },
                tasks, members
            )
            if not ok:
                continue

            fp = _log_fingerprint(r)
            if fp in existing_fps:
                continue
            existing_fps.add(fp)
            new_rows.append({
                "log_id": 0,  # assign later
                "task_id": int(r["task_id"]),
                "member_id": int(r["member_id"]),
                "date": r["date"],
                "hours_spent": float(r["hours_spent"]),
                "note": r.get("note", ""),
                "is_deleted": False
            })
            imported_count += 1

        # move file to processed
        shutil.move(fpath, os.path.join(PROCESSED_DIR, os.path.basename(fpath)))
        processed_count += 1

    if new_rows:
        # Assign fresh sequential IDs to avoid collisions
        next_id = _next_id(logs, "log_id")
        for i, r in enumerate(new_rows):
            r["log_id"] = next_id + i
        merged = pd.concat([logs, pd.DataFrame(new_rows)], ignore_index=True)
        save_logs(merged)
        logs = merged

    recent = logs[~logs["is_deleted"]].sort_values("date", ascending=False).head(20)
    msg = f"✅ Processed {processed_count} inbox file(s), imported {imported_count} new log(s)."
    if imported_count == 0:
        msg = f"ℹ️ Processed {processed_count} inbox file(s). No new logs to import."
    return msg, recent

# -----------------------
# Data loading & seeding
# -----------------------
def seed_sample_data() -> None:
    """Create sample CSVs if missing, with ~5 members, ~12 tasks, time logs."""
    _ensure_dirs()

    seeded = False

    if not os.path.exists(MEMBERS_CSV):
        members = pd.DataFrame([
            {"member_id": 1, "name": "Alice",   "role": "Product Manager", "weekly_capacity_hours": 40, "is_deleted": False},
            {"member_id": 2, "name": "Bob",     "role": "Backend Engineer", "weekly_capacity_hours": 40, "is_deleted": False},
            {"member_id": 3, "name": "Chandra", "role": "Frontend Engineer","weekly_capacity_hours": 32, "is_deleted": False},
            {"member_id": 4, "name": "Deepa",   "role": "Data Analyst",     "weekly_capacity_hours": 40, "is_deleted": False},
            {"member_id": 5, "name": "Eshan",   "role": "QA Engineer",      "weekly_capacity_hours": 35, "is_deleted": False},
        ])
        members.to_csv(MEMBERS_CSV, index=False)
        seeded = True

    if not os.path.exists(TASKS_CSV):
        today = pd.Timestamp.today().normalize()
        # Spread created_at/due_date across recent window
        def days_ago(n): return (today - pd.Timedelta(days=n)).date().isoformat()
        def days_ahead(n): return (today + pd.Timedelta(days=n)).date().isoformat()

        tasks = pd.DataFrame([
            {"task_id": 1, "title": "Define MVP scope", "description": "Write PRD and acceptance criteria",
             "assignee_id": 1, "status": "Done", "priority": "P2", "estimate_hours": 6,
             "created_at": days_ago(21), "due_date": days_ago(14),
             "tags": "planning,prd", "dependencies": "", "is_deleted": False},

            {"task_id": 2, "title": "Design data model", "description": "Members, Tasks, Logs schemas",
             "assignee_id": 4, "status": "Done", "priority": "P1", "estimate_hours": 8,
             "created_at": days_ago(20), "due_date": days_ago(10),
             "tags": "data,schemas", "dependencies": "1", "is_deleted": False},

            {"task_id": 3, "title": "Set up persistence", "description": "CSV I/O and validators",
             "assignee_id": 2, "status": "Review", "priority": "P1", "estimate_hours": 10,
             "created_at": days_ago(14), "due_date": days_ahead(1),
             "tags": "backend,persistence", "dependencies": "2", "is_deleted": False},

            {"task_id": 4, "title": "Build Kanban UI", "description": "Filters & board rendering",
             "assignee_id": 3, "status": "In-Progress", "priority": "P2", "estimate_hours": 12,
             "created_at": days_ago(12), "due_date": days_ahead(3),
             "tags": "frontend,kanban", "dependencies": "2", "is_deleted": False},

            {"task_id": 5, "title": "Time logging form", "description": "Add & list logs",
             "assignee_id": 5, "status": "In-Progress", "priority": "P3", "estimate_hours": 6,
             "created_at": days_ago(11), "due_date": days_ahead(2),
             "tags": "frontend,logs", "dependencies": "3", "is_deleted": False},

            {"task_id": 6, "title": "Utilization reports", "description": "Capacity & charts",
             "assignee_id": 2, "status": "Backlog", "priority": "P1", "estimate_hours": 10,
             "created_at": days_ago(8), "due_date": days_ahead(7),
             "tags": "reports,utilization", "dependencies": "3", "is_deleted": False},

            {"task_id": 7, "title": "Burndown chart", "description": "Remaining hours over time",
             "assignee_id": 4, "status": "Backlog", "priority": "P2", "estimate_hours": 8,
             "created_at": days_ago(8), "due_date": days_ahead(7),
             "tags": "reports,burndown", "dependencies": "3", "is_deleted": False},

            {"task_id": 8, "title": "Throughput metric", "description": "Done per week",
             "assignee_id": 5, "status": "Backlog", "priority": "P3", "estimate_hours": 5,
             "created_at": days_ago(7), "due_date": days_ahead(8),
             "tags": "reports,throughput", "dependencies": "", "is_deleted": False},

            {"task_id": 9, "title": "Cycle time metric", "description": "Created→Done, ignore weekends toggle",
             "assignee_id": 4, "status": "Backlog", "priority": "P3", "estimate_hours": 7,
             "created_at": days_ago(6), "due_date": days_ahead(9),
             "tags": "reports,cycle", "dependencies": "", "is_deleted": False},

            {"task_id": 10, "title": "WIP counter", "description": "Daily WIP line chart",
             "assignee_id": 3, "status": "Backlog", "priority": "P4", "estimate_hours": 4,
             "created_at": days_ago(6), "due_date": days_ahead(10),
             "tags": "reports,wip", "dependencies": "", "is_deleted": False},

            {"task_id": 11, "title": "Import/Export CSV", "description": "Schema validation & preview",
             "assignee_id": 2, "status": "Backlog", "priority": "P2", "estimate_hours": 8,
             "created_at": days_ago(5), "due_date": days_ahead(12),
             "tags": "admin,io", "dependencies": "", "is_deleted": False},

            {"task_id": 12, "title": "Polish validations", "description": "Edge cases & UX toasts",
             "assignee_id": 1, "status": "Backlog", "priority": "P2", "estimate_hours": 6,
             "created_at": days_ago(4), "due_date": days_ahead(12),
             "tags": "quality,validation", "dependencies": "", "is_deleted": False},
        ])
        tasks.to_csv(TASKS_CSV, index=False)
        seeded = True

    if not os.path.exists(LOGS_CSV):
        # Seed a handful of time logs distributed across tasks/members
        rng = np.random.default_rng(42)
        logs = []
        log_id = 1
        # Make daily logs for the past ~2 weeks for some tasks
        base_day = pd.Timestamp.today().normalize() - pd.Timedelta(days=14)
        for d in range(14):
            day = (base_day + pd.Timedelta(days=d)).date().isoformat()
            # 3 logs per day across different tasks
            for (task_id, member_id, hours) in [(1,1,1.0), (2,4,1.5), (3,2,2.0), (4,3,1.0), (5,5,1.0)]:
                if rng.random() < 0.6:  # ~60% of days
                    logs.append({
                        "log_id": log_id,
                        "task_id": task_id,
                        "member_id": member_id,
                        "date": day,
                        "hours_spent": round(float(hours + rng.normal(0, 0.3)), 2),
                        "note": "Auto-seeded",
                        "is_deleted": False
                    })
                    log_id += 1
        pd.DataFrame(logs).to_csv(LOGS_CSV, index=False)
        seeded = True

    return seeded

def load_members() -> pd.DataFrame:
    _ensure_dirs()
    if not os.path.exists(MEMBERS_CSV):
        seed_sample_data()
    df = pd.read_csv(MEMBERS_CSV)
    # Normalize cols
    df["member_id"] = df["member_id"].astype(int)
    df["name"] = df["name"].astype(str)
    df["role"] = df["role"].astype(str)
    df["weekly_capacity_hours"] = df["weekly_capacity_hours"].fillna(40).astype(float)
    df = _normalize_bool_col(df, "is_deleted")
    return df

def load_tasks() -> pd.DataFrame:
    _ensure_dirs()
    if not os.path.exists(TASKS_CSV):
        seed_sample_data()
    df = pd.read_csv(TASKS_CSV)
    # Normalize
    df["task_id"] = df["task_id"].astype(int)
    df["title"] = df["title"].astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["assignee_id"] = df["assignee_id"].fillna(0).astype(int)
    df["status"] = df["status"].astype(str)
    df["priority"] = df["priority"].astype(str)
    df["estimate_hours"] = df["estimate_hours"].fillna(0.0).astype(float)
    df["created_at"] = df["created_at"].apply(_parse_date_safe).dt.date.astype(str)
    df["due_date"] = df["due_date"].apply(_parse_date_safe).dt.date.astype(str)
    df["tags"] = df["tags"].fillna("").astype(str)
    df["dependencies"] = df["dependencies"].fillna("").astype(str)
    df = _normalize_bool_col(df, "is_deleted")
    return df

def load_logs() -> pd.DataFrame:
    _ensure_dirs()
    if not os.path.exists(LOGS_CSV):
        seed_sample_data()
    df = pd.read_csv(LOGS_CSV)
    if df.empty:
        # Ensure columns
        df = pd.DataFrame(columns=["log_id","task_id","member_id","date","hours_spent","note","is_deleted"])
    df["log_id"] = df["log_id"].astype(int)
    df["task_id"] = df["task_id"].astype(int)
    df["member_id"] = df["member_id"].astype(int)
    df["date"] = df["date"].apply(_parse_date_safe).dt.date.astype(str)
    df["hours_spent"] = df["hours_spent"].fillna(0.0).astype(float)
    df["note"] = df["note"].fillna("").astype(str)
    df = _normalize_bool_col(df, "is_deleted")
    return df

def save_members(df: pd.DataFrame):
    df.to_csv(MEMBERS_CSV, index=False)

def save_tasks(df: pd.DataFrame):
    df.to_csv(TASKS_CSV, index=False)

def save_logs(df: pd.DataFrame):
    df.to_csv(LOGS_CSV, index=False)

# ----------------
# Validations
# ----------------
def validate_task(new_task: dict, tasks: pd.DataFrame, members: pd.DataFrame) -> Tuple[bool, str]:
    """Validate task before insert/update."""
    # estimate ≥ 0
    if _safe_float(new_task.get("estimate_hours"), 0.0) < 0:
        return False, "Estimate hours must be ≥ 0."
    # assignee must exist (except 0/unassigned allowed?)
    assignee_id = _safe_int(new_task.get("assignee_id"), 0)
    if assignee_id != 0:
        if not ((members["member_id"] == assignee_id) & (~members["is_deleted"])).any():
            return False, f"Assignee with id={assignee_id} does not exist."
    # status
    if new_task.get("status") not in STATUSES:
        return False, f"Status must be one of {STATUSES}"
    # priority
    if new_task.get("priority") not in PRIORITIES:
        return False, f"Priority must be one of {PRIORITIES}"
    # due_date ≥ created_at
    created = _parse_date_safe(new_task.get("created_at"))
    due = _parse_date_safe(new_task.get("due_date"))
    if pd.isna(created) or pd.isna(due):
        return False, "Created and Due dates must be valid."
    if due < created:
        return False, "Due date must be on or after Created date."
    # dependencies
    dep_ids = _split_csv_ids(new_task.get("dependencies", ""))
    if dep_ids:
        valid_ids = set(tasks["task_id"].tolist())
        bad = [d for d in dep_ids if d not in valid_ids]
        if bad:
            return False, f"Invalid dependency task_id(s): {bad}"
    return True, "OK"

def validate_time_log(new_log: dict, tasks: pd.DataFrame, members: pd.DataFrame) -> Tuple[bool, str]:
    """Validate time log before insert/update."""
    # positive hours
    hours = _safe_float(new_log.get("hours_spent"), 0.0)
    if hours <= 0:
        return False, "Hours must be > 0."
    # task exists and not deleted
    t_id = _safe_int(new_log.get("task_id"), 0)
    if not ((tasks["task_id"] == t_id) & (~tasks["is_deleted"])).any():
        return False, f"Task with id={t_id} does not exist."
    # member exists
    m_id = _safe_int(new_log.get("member_id"), 0)
    if not ((members["member_id"] == m_id) & (~members["is_deleted"])).any():
        return False, f"Member with id={m_id} does not exist."
    # date valid
    if pd.isna(_parse_date_safe(new_log.get("date"))):
        return False, "Date must be valid."
    return True, "OK"

def validate_member(new_member: dict) -> Tuple[bool, str]:
    cap = _safe_float(new_member.get("weekly_capacity_hours"), 0.0)
    if cap <= 0:
        return False, "Weekly capacity hours must be > 0."
    if not str(new_member.get("name","")).strip():
        return False, "Member name cannot be empty."
    return True, "OK"

# ----------------
# Joins & deriveds
# ----------------
def with_assignee_name(tasks: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    m = members[~members["is_deleted"]][["member_id","name"]].rename(columns={"name":"assignee"})
    out = tasks.merge(m, how="left", left_on="assignee_id", right_on="member_id")
    out["assignee"] = out["assignee"].fillna("Unassigned")
    out = out.drop(columns=["member_id"])
    return out

def compute_logged_hours_per_task(logs: pd.DataFrame) -> pd.Series:
    valid = logs[~logs["is_deleted"]]
    if valid.empty:
        return pd.Series(dtype=float)
    return valid.groupby("task_id")["hours_spent"].sum()

def tasks_with_logged_hours(tasks: pd.DataFrame, logs: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    t = with_assignee_name(tasks, members)
    sums = compute_logged_hours_per_task(logs)
    t = t.copy()
    t["logged_hours"] = t["task_id"].map(sums).fillna(0.0)
    t["remaining_hours"] = (t["estimate_hours"] - t["logged_hours"]).clip(lower=0.0)
    return t

# ----------------
# Filtering
# ----------------
def filter_tasks(
    tasks_df: pd.DataFrame,
    members_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    assignees: List[str],
    priorities: List[str],
    tags: List[str],
    start_date: str | None,
    end_date: str | None,
    search: str
) -> pd.DataFrame:
    t = tasks_with_logged_hours(tasks_df[~tasks_df["is_deleted"]], logs_df, members_df)
    # Map assignee names
    m_map = members_df.set_index("member_id")["name"].to_dict()
    t["assignee_name"] = t["assignee_id"].map(m_map).fillna("Unassigned")

    # Filters
    if assignees:
        t = t[t["assignee_name"].isin(assignees)]
    if priorities:
        t = t[t["priority"].isin(priorities)]
    if tags:
        # any overlap with task.tags
        def has_any_tag(tagstr: str) -> bool:
            if not tagstr:
                return False
            task_tags = set([s.strip().lower() for s in tagstr.split(",") if s.strip()])
            return any(tt.lower() in task_tags for tt in tags)
        t = t[t["tags"].apply(has_any_tag)]
    sd = _parse_date_safe(start_date) if start_date else pd.NaT
    ed = _parse_date_safe(end_date) if end_date else pd.NaT
    if not pd.isna(sd) and not pd.isna(ed):
        # Keep tasks whose created_at or due_date overlaps window
        c = t["created_at"].apply(_parse_date_safe)
        d = t["due_date"].apply(_parse_date_safe)
        mask = (c.between(sd, ed)) | (d.between(sd, ed))
        t = t[mask]
    if search and search.strip():
        s = search.strip().lower()
        t = t[
            t["title"].str.lower().str.contains(s) |
            t["description"].str.lower().str.contains(s) |
            t["tags"].str.lower().str.contains(s)
        ]
    return t

# ----------------
# Kanban rendering
# ----------------
def render_kanban_html(filtered_tasks: pd.DataFrame, members_df: pd.DataFrame) -> str:
    """Return a simple responsive HTML Kanban grid."""
    # Group by status in STATUSES order
    filtered_tasks = filtered_tasks.copy()
    filtered_tasks["due_date"] = filtered_tasks["due_date"].fillna("")
    filtered_tasks["assignee"] = filtered_tasks["assignee_name"].fillna("Unassigned") if "assignee_name" in filtered_tasks.columns else filtered_tasks.get("assignee","Unassigned")

    cols = []
    for status in STATUSES:
        col_tasks = filtered_tasks[filtered_tasks["status"] == status].sort_values(["priority","due_date"])
        cards_html = []
        for _, row in col_tasks.iterrows():
            card = f"""
            <div style="border:1px solid #ddd; border-radius:8px; padding:8px; margin-bottom:8px;">
              <div style="font-weight:600;">[{row['priority']}] {row['title']}</div>
              <div style="font-size:12px; margin-top:4px;">
                Assignee: <b>{row.get('assignee','Unassigned')}</b><br/>
                Est: {row['estimate_hours']}h &nbsp;|&nbsp; Logged: {row.get('logged_hours',0):.1f}h<br/>
                Due: {row['due_date']}<br/>
                Tags: {row.get('tags','')}
              </div>
              <div style="font-size:12px; color:#666; margin-top:4px;">Task ID: {row['task_id']}</div>
            </div>
            """
            cards_html.append(card)
        col_html = f"""
        <div style="flex:1; min-width:240px; padding:8px;">
          <div style="font-weight:700; margin-bottom:8px;">{status}</div>
          {''.join(cards_html) if cards_html else '<div style="color:#888;">No tasks</div>'}
        </div>
        """
        cols.append(col_html)

    grid = f"""
    <div style="display:flex; gap:8px; align-items:flex-start; justify-content:stretch; width:100%; overflow-x:auto;">
      {''.join(cols)}
    </div>
    """
    return grid

# ----------------
# Exports
# ----------------
def export_df_to_csv(df: pd.DataFrame, name_prefix: str) -> str:
    """Save df to EXPORT_DIR with timestamp; return file path."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(EXPORT_DIR, f"{name_prefix}_{ts}.csv")
    df.to_csv(path, index=False)
    return path

def export_figure_to_png(fig: plt.Figure, name_prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(EXPORT_DIR, f"{name_prefix}_{ts}.png")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path

# ----------------
# Reports & Metrics
# ----------------
def _date_window(logs: pd.DataFrame, start: str | date, end: str | date) -> pd.DataFrame:
    sd = _parse_date_safe(start)
    ed = _parse_date_safe(end)
    if pd.isna(sd) or pd.isna(ed) or ed < sd:
        return logs.iloc[0:0]
    dates = logs["date"].apply(_parse_date_safe)
    return logs[(dates >= sd) & (dates <= ed)]

def compute_utilization(
    members: pd.DataFrame,
    logs: pd.DataFrame,
    start: str | date,
    end: str | date
) -> pd.DataFrame:
    """Utilization % = (sum hours in window) / (weekly_capacity * (days/7)) * 100"""
    sd = _parse_date_safe(start)
    ed = _parse_date_safe(end)
    if pd.isna(sd) or pd.isna(ed) or ed < sd:
        return pd.DataFrame(columns=["member_id","name","hours","capacity","utilization_pct"])
    days = _date_range_days(sd, ed)
    prorate_factor = days / 7.0

    logs_win = _date_window(logs[~logs["is_deleted"]], sd, ed)
    by_member = logs_win.groupby("member_id")["hours_spent"].sum() if not logs_win.empty else pd.Series(dtype=float)

    rows = []
    for _, m in members[~members["is_deleted"]].iterrows():
        hours = float(by_member.get(m["member_id"], 0.0))
        capacity = float(m["weekly_capacity_hours"]) * prorate_factor
        util = (hours / capacity * 100.0) if capacity > 0 else 0.0
        rows.append({
            "member_id": m["member_id"],
            "name": m["name"],
            "hours": round(hours, 2),
            "capacity": round(capacity, 2),
            "utilization_pct": round(util, 1)
        })
    return pd.DataFrame(rows)

def plot_utilization_bar(util_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()
    if util_df.empty:
        ax.set_title("Utilization (no data)")
        return fig
    ax.bar(util_df["name"], util_df["utilization_pct"])
    ax.set_ylabel("Utilization %")
    ax.set_title("Utilization by Member")
    ax.set_ylim(0, max(100, util_df["utilization_pct"].max() * 1.1))
    for i, v in enumerate(util_df["utilization_pct"]):
        ax.text(i, v, f"{v:.0f}%", ha="center", va="bottom")
    return fig

def heatmap_member_week(logs: pd.DataFrame, members: pd.DataFrame, start: str | date, end: str | date) -> Tuple[plt.Figure, pd.DataFrame]:
    """Return (figure, pivot_df) where pivot_df index=member name, columns=YYYY-WW, hours."""
    logs_win = _date_window(logs[~logs["is_deleted"]], start, end)
    if logs_win.empty:
        fig, ax = plt.subplots()
        ax.set_title("Heatmap (no data)")
        return fig, pd.DataFrame()
    df = logs_win.copy()
    df["week"] = df["date"].apply(_parse_date_safe).dt.isocalendar().week.astype(int)
    df["year"] = df["date"].apply(_parse_date_safe).dt.isocalendar().year.astype(int)
    df["year_week"] = df["year"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2)

    m_map = members.set_index("member_id")["name"].to_dict()
    df["member"] = df["member_id"].map(m_map).fillna("Unknown")

    pivot = df.pivot_table(index="member", columns="year_week", values="hours_spent", aggfunc="sum", fill_value=0.0)
    # Plot
    fig, ax = plt.subplots()
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(list(pivot.columns), rotation=45, ha="right")
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(list(pivot.index))
    ax.set_title("Hours Heatmap (Member × Week)")
    fig.colorbar(im, ax=ax, label="Hours")
    return fig, pivot

def burndown_series(tasks: pd.DataFrame, logs: pd.DataFrame, start: str | date, end: str | date) -> pd.DataFrame:
    """Burndown for tasks with due_date in window:
       remaining = sum(estimate - logged_to_date) across tasks, per day."""
    sd = _parse_date_safe(start)
    ed = _parse_date_safe(end)
    if pd.isna(sd) or pd.isna(ed) or ed < sd:
        return pd.DataFrame(columns=["date","remaining_hours"])
    active = tasks[~tasks["is_deleted"]].copy()
    active["due_ts"] = active["due_date"].apply(_parse_date_safe)
    active = active[(active["due_ts"] >= sd) & (active["due_ts"] <= ed)]
    if active.empty:
        return pd.DataFrame(columns=["date","remaining_hours"])

    # Precompute total estimate per task
    est = active.set_index("task_id")["estimate_hours"].to_dict()
    # Logs within window (but cumulative up to each day needed)
    logs_df = logs[~logs["is_deleted"]].copy()
    logs_df["date_ts"] = logs_df["date"].apply(_parse_date_safe)

    days = pd.date_range(sd, ed, freq="D")
    out = []
    for day in days:
        # logged up to the day
        upto = logs_df[logs_df["date_ts"] <= day].groupby("task_id")["hours_spent"].sum() if not logs_df.empty else pd.Series(dtype=float)
        remaining = 0.0
        for t_id, e in est.items():
            rem = max(0.0, float(e) - float(upto.get(t_id, 0.0)))
            remaining += rem
        out.append({"date": day.date().isoformat(), "remaining_hours": round(remaining, 2)})
    return pd.DataFrame(out)

def plot_burndown(burn_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()
    if burn_df.empty:
        ax.set_title("Burndown (no active sprint tasks)")
        return fig
    x = pd.to_datetime(burn_df["date"])
    y = burn_df["remaining_hours"]
    ax.plot(x, y, marker="o")
    ax.set_title("Burndown (Remaining Hours)")
    ax.set_ylabel("Hours")
    ax.set_xlabel("Date")
    ax.grid(True, which="both", axis="both", linestyle="--", alpha=0.3)
    return fig

def throughput_series(tasks: pd.DataFrame, start: str | date, end: str | date) -> pd.DataFrame:
    """Tasks Done per week (ISO week of due or done date; we use due_date if status=Done)."""
    sd = _parse_date_safe(start); ed = _parse_date_safe(end)
    if pd.isna(sd) or pd.isna(ed) or ed < sd:
        return pd.DataFrame(columns=["year_week","count"])
    t = tasks[~tasks["is_deleted"]].copy()
    t = t[t["status"] == "Done"]
    if t.empty:
        return pd.DataFrame(columns=["year_week","count"])
    t["done_date"] = t["due_date"].apply(_parse_date_safe)  # approximation
    t = t[(t["done_date"] >= sd) & (t["done_date"] <= ed)]
    if t.empty:
        return pd.DataFrame(columns=["year_week","count"])
    t["year_week"] = t["done_date"].dt.isocalendar().year.astype(str) + "-W" + t["done_date"].dt.isocalendar().week.astype(str).str.zfill(2)
    counts = t.groupby("year_week")["task_id"].count().reset_index().rename(columns={"task_id":"count"})
    return counts

def plot_throughput(tp_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()
    if tp_df.empty:
        ax.set_title("Throughput (no Done tasks)")
        return fig
    ax.bar(tp_df["year_week"], tp_df["count"])
    ax.set_title("Throughput (Done per Week)")
    ax.set_ylabel("Count")
    ax.set_xticklabels(tp_df["year_week"], rotation=45, ha="right")
    return fig

def cycle_time_df(tasks: pd.DataFrame, ignore_weekends: bool) -> pd.DataFrame:
    """Compute cycle time in days: created_at → due_date (proxy for done). Only for Done tasks."""
    t = tasks[(~tasks["is_deleted"]) & (tasks["status"] == "Done")].copy()
    if t.empty:
        return pd.DataFrame(columns=["task_id","title","cycle_days"])
    created = t["created_at"].apply(_parse_date_safe)
    done = t["due_date"].apply(_parse_date_safe)  # proxy
    days = []
    for c, d in zip(created, done):
        if pd.isna(c) or pd.isna(d) or d < c:
            days.append(np.nan)
        else:
            if ignore_weekends:
                days.append(_biz_days_between(c, d))
            else:
                days.append(_date_range_days(c, d))
    t["cycle_days"] = days
    return t[["task_id","title","cycle_days"]].dropna()

def plot_cycle_time(ct_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()
    if ct_df.empty:
        ax.set_title("Cycle Time (no Done tasks)")
        return fig
    ax.bar(ct_df["task_id"].astype(str), ct_df["cycle_days"])
    ax.set_title("Cycle Time (Days)")
    ax.set_ylabel("Days")
    ax.set_xlabel("Task ID")
    return fig

def wip_series(tasks: pd.DataFrame, start: str | date, end: str | date) -> pd.DataFrame:
    """Count of tasks in non-Done statuses per day, based on created/due proxy."""
    sd = _parse_date_safe(start); ed = _parse_date_safe(end)
    if pd.isna(sd) or pd.isna(ed) or ed < sd:
        return pd.DataFrame(columns=["date","wip"])
    t = tasks[~tasks["is_deleted"]].copy()
    t["c"] = t["created_at"].apply(_parse_date_safe)
    t["d"] = t["due_date"].apply(_parse_date_safe)
    days = pd.date_range(sd, ed, freq="D")
    out = []
    for day in days:
        # WIP ~ tasks created on/before day and not Done yet (or due in future)
        on = t[(t["c"] <= day) & (t["status"].isin(["Backlog","In-Progress","Review"]))]
        out.append({"date": day.date().isoformat(), "wip": len(on)})
    return pd.DataFrame(out)

def plot_wip(wip_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots()
    if wip_df.empty:
        ax.set_title("WIP (no data)")
        return fig
    x = pd.to_datetime(wip_df["date"])
    y = wip_df["wip"]
    ax.plot(x, y, marker="o")
    ax.set_title("WIP Count by Day")
    ax.set_ylabel("WIP Count")
    ax.set_xlabel("Date")
    ax.grid(True, linestyle="--", alpha=0.3)
    return fig

def quick_pivot(tasks: pd.DataFrame, logs: pd.DataFrame, by: str) -> pd.DataFrame:
    """Pivot counts/hours by category (status/priority/tag/assignee)."""
    t = tasks.copy()
    m = load_members()
    t = with_assignee_name(t, m)
    if by == "status":
        return t.groupby("status")["task_id"].count().reset_index().rename(columns={"task_id":"task_count"})
    if by == "priority":
        return t.groupby("priority")["task_id"].count().reset_index().rename(columns={"task_id":"task_count"})
    if by == "assignee":
        return t.groupby("assignee")["task_id"].count().reset_index().rename(columns={"task_id":"task_count"})
    if by == "tag":
        # explode tags
        t["tags_list"] = t["tags"].apply(lambda s: [x.strip() for x in str(s).split(",") if x.strip()])
        e = t.explode("tags_list")
        if e.empty:
            return pd.DataFrame(columns=["tag","task_count"])
        return e.groupby("tags_list")["task_id"].count().reset_index().rename(columns={"tags_list":"tag","task_id":"task_count"})
    return pd.DataFrame()

# ----------------
# UI Actions (CRUD)
# ----------------
def action_update_status(task_id: int, new_status: str) -> Tuple[str, str]:
    tasks = load_tasks()
    if not ((tasks["task_id"] == task_id) & (~tasks["is_deleted"])).any():
        return "Error", f"Task {task_id} not found."
    if new_status not in STATUSES:
        return "Error", f"Invalid status {new_status}."
    tasks.loc[tasks["task_id"] == task_id, "status"] = new_status
    save_tasks(tasks)
    return "OK", f"Task {task_id} moved to {new_status}"

def action_add_task(title, description, assignee_name, status, priority, estimate, created_at, due_date, tags, dep_text) -> Tuple[str, pd.DataFrame]:
    members = load_members()
    tasks = load_tasks()
    # Map assignee name -> id
    assignee_id = 0
    if assignee_name and assignee_name != "Unassigned":
        row = members[(members["name"] == assignee_name) & (~members["is_deleted"])]
        assignee_id = int(row.iloc[0]["member_id"]) if not row.empty else 0

    new = {
        "task_id": _next_id(tasks, "task_id"),
        "title": title.strip(),
        "description": (description or "").strip(),
        "assignee_id": assignee_id,
        "status": status,
        "priority": priority,
        "estimate_hours": _safe_float(estimate, 0.0),
        "created_at": _parse_date_safe(created_at).date().isoformat() if created_at else _today_str(),
        "due_date": _parse_date_safe(due_date).date().isoformat() if due_date else _today_str(),
        "tags": (tags or "").strip(),
        "dependencies": (dep_text or "").strip(),
        "is_deleted": False
    }
    ok, msg = validate_task(new, tasks, members)
    if not ok:
        return f"❌ {msg}", with_assignee_name(tasks[~tasks["is_deleted"]], members)

    tasks = pd.concat([tasks, pd.DataFrame([new])], ignore_index=True)
    save_tasks(tasks)
    return "✅ Task added", with_assignee_name(tasks[~tasks["is_deleted"]], members)

def action_edit_task(task_id, title, description, assignee_name, status, priority, estimate, created_at, due_date, tags, dep_text) -> Tuple[str, pd.DataFrame]:
    members = load_members()
    tasks = load_tasks()
    if not ((tasks["task_id"] == task_id) & (~tasks["is_deleted"])).any():
        return f"❌ Task {task_id} not found", with_assignee_name(tasks[~tasks["is_deleted"]], members)

    # Map assignee name -> id
    assignee_id = 0
    if assignee_name and assignee_name != "Unassigned":
        row = members[(members["name"] == assignee_name) & (~members["is_deleted"])]
        assignee_id = int(row.iloc[0]["member_id"]) if not row.empty else 0

    new = {
        "task_id": task_id,
        "title": title.strip(),
        "description": (description or "").strip(),
        "assignee_id": assignee_id,
        "status": status,
        "priority": priority,
        "estimate_hours": _safe_float(estimate, 0.0),
        "created_at": _parse_date_safe(created_at).date().isoformat() if created_at else _today_str(),
        "due_date": _parse_date_safe(due_date).date().isoformat() if due_date else _today_str(),
        "tags": (tags or "").strip(),
        "dependencies": (dep_text or "").strip(),
        "is_deleted": False
    }
    ok, msg = validate_task(new, tasks, members)
    if not ok:
        return f"❌ {msg}", with_assignee_name(tasks[~tasks["is_deleted"]], members)

    for k, v in new.items():
        tasks.loc[tasks["task_id"] == task_id, k] = v
    save_tasks(tasks)
    return "✅ Task updated", with_assignee_name(tasks[~tasks["is_deleted"]], members)

def action_delete_task(task_id) -> Tuple[str, pd.DataFrame]:
    members = load_members()
    tasks = load_tasks()
    if not (tasks["task_id"] == task_id).any():
        return f"❌ Task {task_id} not found", with_assignee_name(tasks[~tasks["is_deleted"]], members)
    tasks.loc[tasks["task_id"] == task_id, "is_deleted"] = True
    save_tasks(tasks)
    return "🗑️ Task soft-deleted", with_assignee_name(tasks[~tasks["is_deleted"]], members)

def action_add_log(task_id, member_name, log_date, hours, note) -> Tuple[str, pd.DataFrame]:
    members = load_members()
    tasks = load_tasks()
    logs = load_logs()
    # map member name -> id
    row = members[(members["name"] == member_name) & (~members["is_deleted"])]
    if row.empty:
        return "❌ Member not found", logs[~logs["is_deleted"]].sort_values("date", ascending=False).head(20)

    new = {
        "log_id": _next_id(logs, "log_id"),
        "task_id": int(task_id),
        "member_id": int(row.iloc[0]["member_id"]),
        "date": _parse_date_safe(log_date).date().isoformat() if log_date else _today_str(),
        "hours_spent": _safe_float(hours, 0.0),
        "note": (note or "").strip(),
        "is_deleted": False
    }
    ok, msg = validate_time_log(new, tasks, members)
    if not ok:
        return f"❌ {msg}", logs[~logs["is_deleted"]].sort_values("date", ascending=False).head(20)

    logs = pd.concat([logs, pd.DataFrame([new])], ignore_index=True)
    save_logs(logs)
    return "✅ Log added", logs[~logs["is_deleted"]].sort_values("date", ascending=False).head(20)

def action_edit_log(log_id, task_id, member_name, log_date, hours, note) -> Tuple[str, pd.DataFrame]:
    members = load_members()
    tasks = load_tasks()
    logs = load_logs()
    if not (logs["log_id"] == log_id).any():
        return f"❌ Log {log_id} not found", logs[~logs["is_deleted"]].sort_values("date", ascending=False).head(20)
    # map member name -> id
    row = members[(members["name"] == member_name) & (~members["is_deleted"])]
    if row.empty:
        return "❌ Member not found", logs[~logs["is_deleted"]].sort_values("date", ascending=False).head(20)

    new = {
        "log_id": int(log_id),
        "task_id": int(task_id),
        "member_id": int(row.iloc[0]["member_id"]),
        "date": _parse_date_safe(log_date).date().isoformat() if log_date else _today_str(),
        "hours_spent": _safe_float(hours, 0.0),
        "note": (note or "").strip(),
        "is_deleted": False
    }
    ok, msg = validate_time_log(new, tasks, members)
    if not ok:
        return f"❌ {msg}", logs[~logs["is_deleted"]].sort_values("date", ascending=False).head(20)

    for k, v in new.items():
        logs.loc[logs["log_id"] == log_id, k] = v
    save_logs(logs)
    return "✅ Log updated", logs[~logs["is_deleted"]].sort_values("date", ascending=False).head(20)

def action_delete_log(log_id) -> Tuple[str, pd.DataFrame]:
    logs = load_logs()
    if not (logs["log_id"] == log_id).any():
        return f"❌ Log {log_id} not found", logs[~logs["is_deleted"]].sort_values("date", ascending=False).head(20)
    logs.loc[logs["log_id"] == log_id, "is_deleted"] = True
    save_logs(logs)
    return "🗑️ Log soft-deleted", logs[~logs["is_deleted"]].sort_values("date", ascending=False).head(20)

def action_add_member(name, role, cap) -> Tuple[str, pd.DataFrame]:
    members = load_members()
    new = {
        "member_id": _next_id(members, "member_id"),
        "name": (name or "").strip(),
        "role": (role or "").strip(),
        "weekly_capacity_hours": _safe_float(cap, 40.0),
        "is_deleted": False
    }
    ok, msg = validate_member(new)
    if not ok:
        return f"❌ {msg}", members[~members["is_deleted"]]
    members = pd.concat([members, pd.DataFrame([new])], ignore_index=True)
    save_members(members)
    return "✅ Member added", members[~members["is_deleted"]]

def action_edit_member(member_id, name, role, cap) -> Tuple[str, pd.DataFrame]:
    members = load_members()
    if not (members["member_id"] == member_id).any():
        return f"❌ Member {member_id} not found", members[~members["is_deleted"]]
    new = {
        "member_id": int(member_id),
        "name": (name or "").strip(),
        "role": (role or "").strip(),
        "weekly_capacity_hours": _safe_float(cap, 40.0),
        "is_deleted": False
    }
    ok, msg = validate_member(new)
    if not ok:
        return f"❌ {msg}", members[~members["is_deleted"]]
    for k, v in new.items():
        members.loc[members["member_id"] == member_id, k] = v
    save_members(members)
    return "✅ Member updated", members[~members["is_deleted"]]

def action_delete_member(member_id) -> Tuple[str, pd.DataFrame]:
    members = load_members()
    if not (members["member_id"] == member_id).any():
        return f"❌ Member {member_id} not found", members[~members["is_deleted"]]
    members.loc[members["member_id"] == member_id, "is_deleted"] = True
    save_members(members)
    return "🗑️ Member soft-deleted", members[~members["is_deleted"]]

# ----------------
# Admin import/export
# ----------------
EXPECTED_TASKS_COLS = ["task_id","title","description","assignee_id","status","priority","estimate_hours","created_at","due_date","tags","dependencies","is_deleted"]
EXPECTED_MEMBERS_COLS = ["member_id","name","role","weekly_capacity_hours","is_deleted"]
EXPECTED_LOGS_COLS = ["log_id","task_id","member_id","date","hours_spent","note","is_deleted"]

def _validate_columns(df: pd.DataFrame, expected: List[str]) -> Tuple[bool, str]:
    cols = list(df.columns)
    missing = [c for c in expected if c not in cols]
    if missing:
        return False, f"Missing columns: {missing}"
    return True, "OK"

def import_csv(kind: str, fileobj) -> Tuple[str, pd.DataFrame]:
    """kind: 'tasks' | 'members' | 'logs'"""
    if fileobj is None:
        return "❌ Please upload a CSV file", pd.DataFrame()
    try:
        df = pd.read_csv(fileobj.name)
    except Exception as e:
        return f"❌ Failed to read CSV: {e}", pd.DataFrame()

    if kind == "tasks":
        ok, msg = _validate_columns(df, EXPECTED_TASKS_COLS)
        if not ok:
            return f"❌ {msg}", pd.DataFrame()
        save_tasks(df)
        return "✅ Tasks imported", df
    elif kind == "members":
        ok, msg = _validate_columns(df, EXPECTED_MEMBERS_COLS)
        if not ok:
            return f"❌ {msg}", pd.DataFrame()
        save_members(df)
        return "✅ Members imported", df
    elif kind == "logs":
        ok, msg = _validate_columns(df, EXPECTED_LOGS_COLS)
        if not ok:
            return f"❌ {msg}", pd.DataFrame()
        save_logs(df)
        return "✅ Time logs imported", df
    else:
        return "❌ Unknown kind", pd.DataFrame()

def reset_sample_data() -> str:
    # Keep exports, reset main CSVs
    for p in [MEMBERS_CSV, TASKS_CSV, LOGS_CSV]:
        if os.path.exists(p):
            os.remove(p)
    seed_sample_data()
    return "✅ Sample data reset."


def _date_str(val) -> str:
    """Return ISO date string (YYYY-MM-DD) or empty if invalid."""
    ts = _parse_date_safe(val)
    return "" if pd.isna(ts) else ts.date().isoformat()

def DateInput(label: str, value=None):
    """
    Gradio Date compatibility shim:
    - If DateInput exists, use it.
    - Otherwise use a Textbox with ISO date text.
    """
    default = _today_str() if value is None else _date_str(value)
    if hasattr(gr, "Date"):
        # DateInput can usually take a date string too
        return DateInput(label=label, value=default)
    else:
        return gr.Textbox(label=f"{label} (YYYY-MM-DD)", value=default)



# ----------------
# Gradio UI build
# ----------------
def build_ui():
    first_seed = seed_sample_data()
    members = load_members()
    tasks = load_tasks()
    logs = load_logs()

    member_names = ["Unassigned"] + members[~members["is_deleted"]]["name"].tolist()
    tag_universe = sorted({t.strip().lower()
                           for s in tasks["tags"].dropna().tolist()
                           for t in str(s).split(",") if t.strip()})

    with gr.Blocks(title=APP_NAME) as demo:
        if first_seed:
            gr.Info("Sample data created under ./data/ — happy building!")

        gr.Markdown(f"## 🧭 {APP_NAME}\nSimple Kanban + time tracking + reports for a small team. Data lives in `./data/`.")

        with gr.Tabs():
            # -------------------------
            # Kanban Tab
            # -------------------------
            with gr.Tab("Kanban"):
                with gr.Row():
                    assignee_filter = gr.CheckboxGroup(choices=members[~members["is_deleted"]]["name"].tolist(), label="Filter: Assignee(s)")
                    priority_filter = gr.CheckboxGroup(choices=PRIORITIES, label="Filter: Priority")
                    tag_filter = gr.CheckboxGroup(choices=tag_universe, label="Filter: Tags")
                with gr.Row():
                    start_filter = DateInput(label="Start date")
                    end_filter = DateInput(label="End date")
                    search_filter = gr.Textbox(label="Search (title/desc/tags)")
                    apply_filters = gr.Button("Apply Filters", variant="primary")
                kanban_html = gr.HTML()

                gr.Markdown("### Quick Edit")
                with gr.Row():
                    task_select = gr.Dropdown(choices=[f"{r.task_id} — {r.title}" for _, r in tasks[~tasks["is_deleted"]].iterrows()],
                                              label="Select Task")
                    new_status = gr.Dropdown(choices=STATUSES, value="In-Progress", label="New Status")
                    move_btn = gr.Button("Update Status")
                move_status_msg = gr.Markdown()

                export_filtered_btn = gr.Button("Export current tasks CSV")
                export_filtered_file = gr.File(label="Download exported tasks CSV", interactive=False)

                def refresh_kanban(assignees, priorities, tags, sd, ed, s):
                    t = load_tasks()
                    m = load_members()
                    l = load_logs()
                    filtered = filter_tasks(t, m, l, assignees or [], priorities or [], tags or [], sd, ed, s or "")
                    html = render_kanban_html(filtered, m)
                    # also refresh task dropdown
                    dd_choices = [f"{r.task_id} — {r.title}" for _, r in filtered.iterrows()] or [f"{r.task_id} — {r.title}" for _, r in t[~t["is_deleted"]].iterrows()]
                    return html, gr.Dropdown(choices=dd_choices, value=None)

                apply_filters.click(
                    refresh_kanban,
                    inputs=[assignee_filter, priority_filter, tag_filter, start_filter, end_filter, search_filter],
                    outputs=[kanban_html, task_select]
                )

                def do_move(selected, status):
                    if not selected:
                        # Re-render board with current filters; return raw HTML string (no .update())
                        t = load_tasks(); m = load_members(); l = load_logs()
                        filtered = filter_tasks(
                            t, m, l,
                            assignee_filter.value or [], priority_filter.value or [], tag_filter.value or [],
                            start_filter.value, end_filter.value, search_filter.value or ""
                        )
                        html = render_kanban_html(filtered, m)
                        return "❌ Select a task first", html

                    task_id = int(str(selected).split("—")[0].strip())
                    code, msg = action_update_status(task_id, status)

                    # Refresh Kanban after move and return raw HTML (string) instead of gr.HTML.update(...)
                    t = load_tasks(); m = load_members(); l = load_logs()
                    filtered = filter_tasks(
                        t, m, l,
                        assignee_filter.value or [], priority_filter.value or [], tag_filter.value or [],
                        start_filter.value, end_filter.value, search_filter.value or ""
                    )
                    html = render_kanban_html(filtered, m)

                    return f"{'✅' if code=='OK' else '❌'} {msg}", html

                move_btn.click(
                    do_move,
                    inputs=[task_select, new_status],
                    outputs=[move_status_msg, kanban_html]
                )

                def export_current():
                    # Export whatever the current filtered view would be
                    t = load_tasks(); m = load_members(); l = load_logs()
                    filtered = filter_tasks(
                        t, m, l,
                        assignee_filter.value or [], priority_filter.value or [], tag_filter.value or [],
                        start_filter.value, end_filter.value, search_filter.value or ""
                    )
                    path = export_df_to_csv(filtered, "tasks_filtered")
                    return gr.update(value=path, visible=True)

                export_filtered_btn.click(export_current, outputs=[export_filtered_file])

            # -------------------------
            # Tasks Tab (CRUD)
            # -------------------------
            with gr.Tab("Tasks"):
                with gr.Row():
                    tasks_table = gr.Dataframe(
                        value=with_assignee_name(load_tasks()[~load_tasks()["is_deleted"]], load_members()),
                        label="Tasks (read-only)",
                        interactive=False
                    )
                gr.Markdown("### Add Task")
                with gr.Row():
                    t_title = gr.Textbox(label="Title")
                    t_assignee = gr.Dropdown(choices=member_names, value="Unassigned", label="Assignee")
                    t_status = gr.Dropdown(choices=STATUSES, value="Backlog", label="Status")
                    t_priority = gr.Dropdown(choices=PRIORITIES, value="P3", label="Priority")
                with gr.Row():
                    t_est = gr.Number(label="Estimate (hours)", value=4.0, precision=2)
                    t_created = DateInput(label="Created At", value=date.today())
                    t_due = DateInput(label="Due Date", value=date.today()+timedelta(days=7))
                with gr.Row():
                    t_tags = gr.Textbox(label="Tags (comma-separated)")
                    t_deps = gr.Textbox(label="Dependencies (task_id comma-separated)")
                t_desc = gr.Textbox(label="Description", lines=3)
                add_task_btn = gr.Button("Add Task", variant="primary")
                add_task_msg = gr.Markdown()

                # def add_task_cb(*args):
                #     msg, tbl = action_add_task(*args)
                #     return msg, tbl

                # add_task_btn.click(
                #     add_task_cb,
                #     inputs=[t_title, t_desc, t_assignee, t_status, t_priority, t_est, t_created, t_due, t_tags, t_deps],
                #     outputs=[add_task_msg, tasks_table]
                # )

                # --- replace existing add_task_cb + click with this version ---

                


                gr.Markdown("### Edit / Delete Task")
                task_pick = gr.Dropdown(choices=[int(r.task_id) for _, r in load_tasks()[~load_tasks()["is_deleted"]].iterrows()],
                                        label="Select Task ID")
                with gr.Row():
                    e_title = gr.Textbox(label="Title")
                    e_assignee = gr.Dropdown(choices=member_names, value="Unassigned", label="Assignee")
                    e_status = gr.Dropdown(choices=STATUSES, value="Backlog", label="Status")
                    e_priority = gr.Dropdown(choices=PRIORITIES, value="P3", label="Priority")
                with gr.Row():
                    e_est = gr.Number(label="Estimate (hours)", value=4.0, precision=2)
                    e_created = DateInput(label="Created At", value=date.today())
                    e_due = DateInput(label="Due Date", value=date.today())
                with gr.Row():
                    e_tags = gr.Textbox(label="Tags")
                    e_deps = gr.Textbox(label="Dependencies")
                e_desc = gr.Textbox(label="Description", lines=3)
                with gr.Row():
                    save_edit_btn = gr.Button("Save Changes", variant="primary")
                    delete_task_btn = gr.Button("Soft Delete", variant="secondary")
                edit_task_msg = gr.Markdown()

                def load_task_fields(task_id):
                    t = load_tasks()
                    m = load_members()
                    row = t[t["task_id"] == int(task_id)]
                    if row.empty:
                        return ("❌ Task not found",) + (gr.update(),)*10
                    r = row.iloc[0].to_dict()
                    # map assignee id -> name
                    name = "Unassigned"
                    if r["assignee_id"] != 0:
                        mm = m[m["member_id"] == int(r["assignee_id"])]
                        if not mm.empty:
                            name = mm.iloc[0]["name"]
                    return (
                        f"Loaded Task {task_id}",
                        gr.update(value=r["title"]),
                        gr.update(value=name),
                        gr.update(value=r["status"]),
                        gr.update(value=r["priority"]),
                        gr.update(value=float(r["estimate_hours"])),
                        gr.update(value=_parse_date_safe(r["created_at"]).date()),
                        gr.update(value=_parse_date_safe(r["due_date"]).date()),
                        gr.update(value=r["tags"]),
                        gr.update(value=r["dependencies"]),
                        gr.update(value=r["description"])
                    )

                task_pick.change(
                    load_task_fields,
                    inputs=[task_pick],
                    outputs=[edit_task_msg, e_title, e_assignee, e_status, e_priority, e_est, e_created, e_due, e_tags, e_deps, e_desc]
                )

                # def save_edit():
                #     msg, tbl = action_edit_task(
                #         task_pick.value, e_title.value, e_desc.value, e_assignee.value, e_status.value,
                #         e_priority.value, e_est.value, e_created.value, e_due.value, e_tags.value, e_deps.value
                #     )
                #     # refresh task id dropdown too
                #     ids = [int(r.task_id) for _, r in load_tasks()[~load_tasks()["is_deleted"]].iterrows()]
                #     return msg, tbl, gr.update(choices=ids)

                # save_edit_btn.click(
                #     save_edit,
                #     outputs=[edit_task_msg, tasks_table, task_pick]
                # )
                # --- replace save_edit() with this version ---

                # def del_task():
                #     msg, tbl = action_delete_task(task_pick.value)
                #     ids = [int(r.task_id) for _, r in load_tasks()[~load_tasks()["is_deleted"]].iterrows()]
                #     return msg, tbl, gr.update(choices=ids, value=None)

                # delete_task_btn.click(
                #     del_task,
                #     outputs=[edit_task_msg, tasks_table, task_pick]
                # )
                # --- replace del_task() with this version ---

            # -------------------------
            # Time Logs Tab
            # -------------------------
            with gr.Tab("Time Logs"):
                with gr.Row():
                    tl_task = gr.Dropdown(choices=[f"{r.task_id} — {r.title}" for _, r in load_tasks()[~load_tasks()["is_deleted"]].iterrows()],
                                          label="Task")
                    tl_member = gr.Dropdown(choices=members[~members["is_deleted"]]["name"].tolist(), label="Member")
                with gr.Row():
                    tl_date = DateInput(label="Date", value=date.today())
                    tl_hours = gr.Number(label="Hours", value=1.0, precision=2)
                tl_note = gr.Textbox(label="Note", lines=2)
                add_log_btn = gr.Button("Add Log", variant="primary")
                add_log_msg = gr.Markdown()
                refresh_task_lists_btn = gr.Button("Refresh Tasks")
                recent_logs = gr.Dataframe(
                    value=load_logs()[~load_logs()["is_deleted"]].sort_values("date", ascending=False).head(20),
                    label="Recent Logs",
                    interactive=False
                )
                

                def add_log_cb(task_label, member_name, d, h, note):
                    if not task_label:
                        return "❌ Select a task", recent_logs.value
                    task_id = int(str(task_label).split("—")[0].strip())
                    msg, df = action_add_log(task_id, member_name, d, h, note)
                    return msg, df

                add_log_btn.click(
                    add_log_cb,
                    inputs=[tl_task, tl_member, tl_date, tl_hours, tl_note],
                    outputs=[add_log_msg, recent_logs]
                )

                gr.Markdown("### Edit / Delete Log")
                log_pick = gr.Dropdown(choices=[int(r.log_id) for _, r in load_logs()[~load_logs()["is_deleted"]].iterrows()],
                                       label="Select Log ID")
                with gr.Row():
                    el_task = gr.Dropdown(choices=[int(r.task_id) for _, r in load_tasks()[~load_tasks()["is_deleted"]].iterrows()], label="Task ID")
                    el_member = gr.Dropdown(choices=members[~members["is_deleted"]]["name"].tolist(), label="Member")
                with gr.Row():
                    el_date = DateInput(label="Date")
                    el_hours = gr.Number(label="Hours", precision=2)
                el_note = gr.Textbox(label="Note")
                with gr.Row():
                    save_log_btn = gr.Button("Save Changes", variant="primary")
                    delete_log_btn = gr.Button("Soft Delete", variant="secondary")
                edit_log_msg = gr.Markdown()

                def load_log_fields(log_id):
                    logs = load_logs()
                    row = logs[logs["log_id"] == int(log_id)]
                    if row.empty:
                        return ("❌ Log not found",) + (gr.update(),)*5
                    r = row.iloc[0]
                    return (
                        f"Loaded Log {log_id}",
                        gr.update(value=int(r["task_id"])),
                        gr.update(value=load_members().set_index("member_id").loc[int(r["member_id"]), "name"] if int(r["member_id"]) in load_members()["member_id"].tolist() else None),
                        gr.update(value=_parse_date_safe(r["date"]).date()),
                        gr.update(value=float(r["hours_spent"])),
                        gr.update(value=r["note"])
                    )

                log_pick.change(
                    load_log_fields,
                    inputs=[log_pick],
                    outputs=[edit_log_msg, el_task, el_member, el_date, el_hours, el_note]
                )

                def save_log():
                    msg, df = action_edit_log(
                        log_pick.value, el_task.value, el_member.value, el_date.value, el_hours.value, el_note.value
                    )
                    ids = [int(r.log_id) for _, r in load_logs()[~load_logs()["is_deleted"]].iterrows()]
                    return msg, df, gr.update(choices=ids)

                save_log_btn.click(
                    save_log,
                    outputs=[edit_log_msg, recent_logs, log_pick]
                )

                def del_log():
                    msg, df = action_delete_log(log_pick.value)
                    ids = [int(r.log_id) for _, r in load_logs()[~load_logs()["is_deleted"]].iterrows()]
                    return msg, df, gr.update(choices=ids, value=None)

                delete_log_btn.click(
                    del_log,
                    outputs=[edit_log_msg, recent_logs, log_pick]
                )

                gr.Markdown("### Per-Task Totals")
                totals_df = gr.Dataframe(
                    value=tasks_with_logged_hours(load_tasks()[~load_tasks()["is_deleted"]], load_logs(), load_members())[["task_id","title","assignee_id","status","priority","estimate_hours","logged_hours","remaining_hours"]],
                    interactive=False
                )

                refresh_totals = gr.Button("Refresh Totals")
                refresh_totals.click(
                    lambda: tasks_with_logged_hours(load_tasks()[~load_tasks()["is_deleted"]], load_logs(), load_members())[["task_id","title","assignee_id","status","priority","estimate_hours","logged_hours","remaining_hours"]],
                    outputs=[totals_df]
                )

            # -------------------------
            # Reports Tab
            # -------------------------
            with gr.Tab("Reports"):
                gr.Markdown("### Window & Options")
                with gr.Row():
                    r_start = DateInput(label="Start date", value=date.today()-timedelta(days=14))
                    r_end = DateInput(label="End date", value=date.today())
                    ignore_wknd = gr.Checkbox(label="Cycle time: ignore weekends", value=True)
                refresh_reports = gr.Button("Refresh Reports", variant="primary")

                gr.Markdown("### Utilization by Member")
                util_plot = gr.Plot()
                util_table = gr.Dataframe(interactive=False)
                util_dl = gr.File(label="Download Utilization PNG", interactive=False)

                gr.Markdown("### Heatmap (Member × Week)")
                heat_plot = gr.Plot()
                heat_table = gr.Dataframe(interactive=False)
                heat_dl = gr.File(label="Download Heatmap PNG", interactive=False)

                gr.Markdown("### Burndown (Tasks due in window)")
                burn_plot = gr.Plot()
                burn_table = gr.Dataframe(interactive=False)
                burn_dl = gr.File(label="Download Burndown PNG", interactive=False)

                gr.Markdown("### Throughput (Done per week)")
                tp_plot = gr.Plot()
                tp_table = gr.Dataframe(interactive=False)
                tp_dl = gr.File(label="Download Throughput PNG", interactive=False)

                gr.Markdown("### Cycle Time (Days)")
                ct_plot = gr.Plot()
                ct_table = gr.Dataframe(interactive=False)
                ct_dl = gr.File(label="Download Cycle Time PNG", interactive=False)

                gr.Markdown("### WIP Count by Day")
                wip_plot = gr.Plot()
                wip_table = gr.Dataframe(interactive=False)
                wip_dl = gr.File(label="Download WIP PNG", interactive=False)

                gr.Markdown("### Quick Pivot")
                pivot_by = gr.Dropdown(choices=["status","priority","tag","assignee"], value="status", label="Group by")
                pivot_table = gr.Dataframe(interactive=False)
                export_metrics_btn = gr.Button("Export Metrics CSV")
                export_metrics_file = gr.File(label="Download metrics CSV", interactive=False)

                def refresh_all_reports(sd, ed, ignore_weekends_flag, group_by):
                    mem = load_members()
                    t = load_tasks()
                    l = load_logs()

                    # Utilization
                    util = compute_utilization(mem, l, sd, ed)
                    f1 = plot_utilization_bar(util)
                    p1 = export_figure_to_png(f1, "utilization")

                    # Heatmap
                    f2, pvt = heatmap_member_week(l, mem, sd, ed)
                    p2 = export_figure_to_png(f2, "heatmap")

                    # Burndown
                    burn = burndown_series(t, l, sd, ed)
                    f3 = plot_burndown(burn)
                    p3 = export_figure_to_png(f3, "burndown")

                    # Throughput
                    th = throughput_series(t, sd, ed)
                    f4 = plot_throughput(th)
                    p4 = export_figure_to_png(f4, "throughput")

                    # Cycle time
                    ct = cycle_time_df(t, ignore_weekends_flag)
                    f5 = plot_cycle_time(ct)
                    p5 = export_figure_to_png(f5, "cycletime")

                    # WIP
                    wip = wip_series(t, sd, ed)
                    f6 = plot_wip(wip)
                    p6 = export_figure_to_png(f6, "wip")

                    # Pivot
                    pv = quick_pivot(t[~t["is_deleted"]], l[~l["is_deleted"]], group_by)

                    return (
                        f1, util, gr.update(value=p1, visible=True),
                        f2, pvt, gr.update(value=p2, visible=True),
                        f3, burn, gr.update(value=p3, visible=True),
                        f4, th, gr.update(value=p4, visible=True),
                        f5, ct, gr.update(value=p5, visible=True),
                        f6, wip, gr.update(value=p6, visible=True),
                        pv
                    )

                refresh_reports.click(
                    refresh_all_reports,
                    inputs=[r_start, r_end, ignore_wknd, pivot_by],
                    outputs=[
                        util_plot, util_table, util_dl,
                        heat_plot, heat_table, heat_dl,
                        burn_plot, burn_table, burn_dl,
                        tp_plot, tp_table, tp_dl,
                        ct_plot, ct_table, ct_dl,
                        wip_plot, wip_table, wip_dl,
                        pivot_table
                    ]
                )

                def export_metrics_csv():
                    mem = load_members()
                    t = load_tasks()
                    l = load_logs()
                    util = compute_utilization(mem, l, r_start.value, r_end.value)
                    path = export_df_to_csv(util, "metrics_utilization")
                    return gr.update(value=path, visible=True)

                export_metrics_btn.click(export_metrics_csv, outputs=[export_metrics_file])

            # -------------------------
            # Admin Tab
            # -------------------------
            with gr.Tab("Admin"):
                gr.Markdown("### Members")
                members_table = gr.Dataframe(value=load_members()[~load_members()["is_deleted"]],
                                             interactive=False)

                with gr.Row():
                    m_name = gr.Textbox(label="Name")
                    m_role = gr.Textbox(label="Role")
                    m_cap = gr.Number(label="Weekly Capacity (hrs)", value=40.0, precision=1)
                    add_member_btn = gr.Button("Add Member", variant="primary")
                add_member_msg = gr.Markdown()

                # --- replace the existing add_mem + click binding ---

                def add_mem(name, role, cap):
                    msg, df = action_add_member(name, role, cap)
                    new_names = df["name"].tolist()
                    return msg, df, gr.update(choices=new_names)

                add_member_btn.click(
                    add_mem,
                    inputs=[m_name, m_role, m_cap],
                    outputs=[add_member_msg, members_table, assignee_filter]
                )


                gr.Markdown("#### Edit / Delete Member")
                m_pick = gr.Dropdown(choices=[int(r.member_id) for _, r in load_members()[~load_members()["is_deleted"]].iterrows()], label="Member ID")
                with gr.Row():
                    em_name = gr.Textbox(label="Name")
                    em_role = gr.Textbox(label="Role")
                    em_cap = gr.Number(label="Weekly Capacity (hrs)", precision=1)
                with gr.Row():
                    save_member_btn = gr.Button("Save", variant="primary")
                    del_member_btn = gr.Button("Soft Delete", variant="secondary")
                edit_member_msg = gr.Markdown()

                # --- replace load_member_fields to handle None safely ---
                def load_member_fields(mid):
                    if mid is None or str(mid).strip() == "":
                        return (
                            "ℹ️ Select a Member ID to load.",
                            gr.update(value=""),
                            gr.update(value=""),
                            gr.update(value=40.0),
                        )
                    df = load_members()
                    try:
                        mid_int = int(mid)
                    except Exception:
                        return (
                            "❌ Invalid Member ID.",
                            gr.update(value=""),
                            gr.update(value=""),
                            gr.update(value=40.0),
                        )
                    row = df[df["member_id"] == mid_int]
                    if row.empty:
                        return (
                            f"❌ Member {mid_int} not found.",
                            gr.update(value=""),
                            gr.update(value=""),
                            gr.update(value=40.0),
                        )
                    r = row.iloc[0]
                    return (
                        f"Loaded Member {mid_int}",
                        gr.update(value=r["name"]),
                        gr.update(value=r["role"]),
                        gr.update(value=float(r["weekly_capacity_hours"])),
                    )

                m_pick.change(
                    load_member_fields,
                    inputs=[m_pick],
                    outputs=[edit_member_msg, em_name, em_role, em_cap],
                )

                def save_member(mid, name, role, cap):
                    if mid is None or str(mid).strip() == "":
                        return "❌ Select a Member ID first", load_members()[~load_members()["is_deleted"]], gr.update()
                    try:
                        mid_int = int(mid)
                    except Exception:
                        return "❌ Invalid Member ID.", load_members()[~load_members()["is_deleted"]], gr.update()

                    msg, df = action_edit_member(mid_int, name, role, cap)
                    ids = [int(r.member_id) for _, r in load_members()[~load_members()["is_deleted"]].iterrows()]
                    return msg, df, gr.update(choices=ids, value=mid_int if msg.startswith("✅") else None)

                save_member_btn.click(
                    save_member,
                    inputs=[m_pick, em_name, em_role, em_cap],
                    outputs=[edit_member_msg, members_table, m_pick],
                )

                def del_member(mid):
                    if mid is None or str(mid).strip() == "":
                        return "❌ Select a Member ID first", load_members()[~load_members()["is_deleted"]], gr.update()
                    try:
                        mid_int = int(mid)
                    except Exception:
                        return "❌ Invalid Member ID.", load_members()[~load_members()["is_deleted"]], gr.update()

                    msg, df = action_delete_member(mid_int)
                    ids = [int(r.member_id) for _, r in load_members()[~load_members()["is_deleted"]].iterrows()]
                    return msg, df, gr.update(choices=ids, value=None)

                del_member_btn.click(
                    del_member,
                    inputs=[m_pick],
                    outputs=[edit_member_msg, members_table, m_pick],
                )

                gr.Markdown("### Import / Export CSV")
                with gr.Row():
                    exp_tasks_btn = gr.Button("Export All Tasks CSV")
                    exp_logs_btn = gr.Button("Export All Time Logs CSV")
                    exp_members_btn = gr.Button("Export All Members CSV")
                with gr.Row():
                    exp_tasks_file = gr.File(label="Download Tasks CSV", interactive=False)
                    exp_logs_file = gr.File(label="Download Logs CSV", interactive=False)
                    exp_members_file = gr.File(label="Download Members CSV", interactive=False)

                exp_tasks_btn.click(lambda: gr.update(value=export_df_to_csv(load_tasks(), "tasks_all"), visible=True), outputs=[exp_tasks_file])
                exp_logs_btn.click(lambda: gr.update(value=export_df_to_csv(load_logs(), "logs_all"), visible=True), outputs=[exp_logs_file])
                exp_members_btn.click(lambda: gr.update(value=export_df_to_csv(load_members(), "members_all"), visible=True), outputs=[exp_members_file])

                gr.Markdown("#### Import (Replace current dataset)")
                with gr.Row():
                    imp_tasks = gr.File(label="Import Tasks CSV")
                    imp_logs = gr.File(label="Import Time Logs CSV")
                    imp_members = gr.File(label="Import Members CSV")
                with gr.Row():
                    imp_tasks_btn = gr.Button("Import Tasks")
                    imp_logs_btn = gr.Button("Import Logs")
                    imp_members_btn = gr.Button("Import Members")
                with gr.Row():
                    imp_tasks_msg = gr.Markdown()
                    imp_logs_msg = gr.Markdown()
                    imp_members_msg = gr.Markdown()

                imp_tasks_btn.click(lambda f: import_csv("tasks", f), inputs=[imp_tasks], outputs=[imp_tasks_msg, tasks_table])
                imp_logs_btn.click(lambda f: import_csv("logs", f), inputs=[imp_logs], outputs=[imp_logs_msg, recent_logs])
                imp_members_btn.click(lambda f: import_csv("members", f), inputs=[imp_members], outputs=[imp_members_msg, members_table])

                gr.Markdown("### Reset Sample Data")
                reset_btn = gr.Button("Reset to Sample Data (keeps exports)", variant="stop")
                reset_msg = gr.Markdown()
                def do_reset():
                    msg = reset_sample_data()
                    return msg, with_assignee_name(load_tasks()[~load_tasks()["is_deleted"]], load_members()), load_members()[~load_members()["is_deleted"]]
                reset_btn.click(do_reset, outputs=[reset_msg, tasks_table, members_table])

                gr.Markdown("### Shared Inbox (Team Sync for Time Logs)")
                gr.Markdown("""
                Use this if everyone runs locally but you want **one consolidated report**.
                - **Push**: exports your time logs to a shared folder (inbox).
                - **Pull & Merge**: the project owner imports all pushed logs into the master board.
                Configure shared folder via env var `KANBAN_INBOX_DIR` (e.g., a network share).
                """)

                # For syncing tasks real time.
                # --- Place this AFTER tl_task, task_select, task_pick, el_task are defined ---

                def add_task_cb(*args):
                    msg, tbl = action_add_task(*args)
                    # Refresh all task-related dropdowns
                    return (
                        msg,
                        tbl,
                        gr.update(choices=_task_label_choices()),   # tl_task
                        gr.update(choices=_task_label_choices()),   # task_select (Kanban quick edit)
                        gr.update(choices=_task_id_choices()),      # task_pick (Tasks -> Edit)
                        gr.update(choices=_task_id_choices()),      # el_task (Time Logs -> Edit)
                    )

                add_task_btn.click(
                    add_task_cb,
                    inputs=[t_title, t_desc, t_assignee, t_status, t_priority, t_est, t_created, t_due, t_tags, t_deps],
                    outputs=[add_task_msg, tasks_table, tl_task, task_select, task_pick, el_task]
                )

                def save_edit():
                    msg, tbl = action_edit_task(
                        task_pick.value, e_title.value, e_desc.value, e_assignee.value, e_status.value,
                        e_priority.value, e_est.value, e_created.value, e_due.value, e_tags.value, e_deps.value
                    )
                    ids = _task_id_choices()
                    return (
                        msg,
                        tbl,
                        gr.update(choices=ids),                   # task_pick
                        gr.update(choices=_task_label_choices()), # tl_task
                        gr.update(choices=_task_label_choices()), # task_select
                        gr.update(choices=ids),                   # el_task
                    )

                save_edit_btn.click(
                    save_edit,
                    outputs=[edit_task_msg, tasks_table, task_pick, tl_task, task_select, el_task]
                )

                def del_task():
                    msg, tbl = action_delete_task(task_pick.value)
                    ids = _task_id_choices()
                    labels = _task_label_choices()
                    return (
                        msg,
                        tbl,
                        gr.update(choices=ids, value=None),  # task_pick
                        gr.update(choices=labels),           # tl_task
                        gr.update(choices=labels),           # task_select
                        gr.update(choices=ids, value=None),  # el_task
                    )

                delete_task_btn.click(
                    del_task,
                    outputs=[edit_task_msg, tasks_table, task_pick, tl_task, task_select, el_task]
                )

                def _refresh_task_lists():
                    return (
                        gr.update(choices=_task_label_choices()),  # tl_task
                        gr.update(choices=_task_id_choices()),     # el_task
                    )

                refresh_task_lists_btn.click(_refresh_task_lists, outputs=[tl_task, el_task])





                # Who's pushing? (used in filename) — defaults to OS username if available
                try:
                    _default_user = os.getlogin()
                except Exception:
                    _default_user = ""
                inbox_user = gr.Textbox(label="Your name/initials for push filename", value=_default_user)

                with gr.Row():
                    push_btn = gr.Button("Push my logs to shared inbox")
                    pull_btn = gr.Button("Pull & merge logs from inbox")

                inbox_msg = gr.Markdown()
                inbox_preview = gr.Dataframe(label="Recent Logs after Merge", interactive=False)

                def do_push(user):
                    return push_my_logs_to_inbox(user)

                def do_pull():
                    msg, df = pull_and_merge_inbox_logs()
                    return msg, df

                push_btn.click(do_push, inputs=[inbox_user], outputs=[inbox_msg])
                pull_btn.click(do_pull, outputs=[inbox_msg, inbox_preview])

            # -------------------------
            # Help / Overview Tab
            # -------------------------
            with gr.Tab("Help / Overview"):
                gr.Markdown(f"""
                # 🧭 Welcome to {APP_NAME}

                This is your **lightweight team Kanban + utilization tracker**.  
                Data is stored locally in `./data/` as CSV files, so it persists between restarts.

                ---

                ## 📘 Tabs Explained

                ### 1. Kanban
                - **What it represents:** A visual board with tasks in Backlog → In-Progress → Review → Done.  
                - **How to use:** Filter tasks by assignee, priority, tag, date range, or search. Update task status via *Quick Edit*.  
                - **What not to do:** Don’t manually edit `tasks.csv` while the app is running—it can cause conflicts.

                ### 2. Tasks
                - **Represents:** Full task list with estimates, due dates, dependencies, etc.  
                - **How to use:** Create new tasks, edit details, or soft-delete old ones. Dependencies should reference valid task IDs.  
                - **What not to do:** Avoid deleting members who still own tasks; instead reassign tasks first.

                ### 3. Time Logs
                - **Represents:** All hours logged against tasks.  
                - **How to use:** Add a log by selecting a task + member, date, and hours. Recent logs and per-task totals help track progress.  
                - **What not to do:** Don’t log 0 or negative hours (validation will block you).

                ### 4. Reports
                - **Represents:** Metrics & charts for utilization, throughput, burndown, cycle time, and WIP.  
                - **How to use:** Pick a date window, refresh reports, and download PNG/CSV outputs. Toggle “ignore weekends” for cycle time.  
                - **What not to do:** Don’t expect reports to auto-update—always click **Refresh Reports** after changes.

                ### 5. Admin
                - **Represents:** System settings and dataset management.  
                - **How to use:** Add/edit members, change weekly capacities, import/export CSVs, reset to sample data, or use the shared inbox to merge logs.  
                - **What not to do:** Avoid resetting sample data unless you want to wipe your current board.

                ---

                ## ⚙️ Technical Flow

                ```mermaid
                flowchart TD
                    UI[Gradio UI Tabs] -->|CRUD Actions| CSV[(members.csv / tasks.csv / time_logs.csv)]
                    CSV -->|Reloads| UI
                    CSV --> Reports[Reports & Charts]
                    UI --> Exports[CSV/PNG Exports]
                    Admin --> Inbox[(Shared Inbox Folder)]
                    Inbox --> CSV
                ```

                - **UI Tabs**: Your interaction points.  
                - **CSV Files**: Persistent store in `./data/`.  
                - **Reports**: Read directly from CSVs each refresh.  
                - **Shared Inbox**: Optional way to merge logs from teammates.  
                - **Exports**: Snapshots of current views.

                ---

                ## 📊 Understanding Reports

                **Utilization by Member**  
                - Shows % of logged hours vs. available capacity.  
                - Formula: `sum(hours_logged) / (weekly_capacity * days/7) * 100`.  
                - Use to balance workload; capacity is pro-rated for partial weeks.

                **Heatmap (Member × Week)**  
                - Visualizes weekly logged hours per member.  
                - Darker cells = heavier load. Spot imbalances quickly.

                **Burndown (Remaining Hours)**  
                - Tracks remaining effort for tasks due in the selected window.  
                - Good = steady decline; Flat = stalled; Spikes = scope change.

                **Throughput**  
                - Tasks moved to *Done* per week.  
                - Shows delivery pace and trend.

                **Cycle Time**  
                - Days from *Created → Done* (toggle to exclude weekends).  
                - Lower is better; investigate long outliers.

                **WIP Count by Day**  
                - Number of active tasks (not Done) daily.  
                - Helps monitor flow; too high WIP = slower cycle times.

                **Quick Pivots**  
                - Snap summaries by status, priority, tag, assignee.  
                - Useful for triage and spotting hot spots.

                ✅ **Tips for accurate reports:**  
                - Always set **Due Dates** on tasks you want in burndown.  
                - Log hours **frequently** (daily preferred).  
                - Refresh reports after imports or edits.  
                - Don’t edit CSVs manually while the app is running.

                ---

                ## 🛠️ Advanced Notes

                - **Data Schemas**  
                * members.csv → `member_id, name, role, weekly_capacity_hours, is_deleted`  
                * tasks.csv → `task_id, title, description, assignee_id, status, priority, estimate_hours, created_at, due_date, tags, dependencies, is_deleted`  
                * time_logs.csv → `log_id, task_id, member_id, date, hours_spent, note, is_deleted`

                - **IDs & References**  
                * Members and tasks are referenced by numeric IDs.  
                * Names can change safely because joins always use IDs.  
                * Dependencies in tasks are stored as comma-separated task IDs.

                - **Validation Rules**  
                * Task estimates must be ≥ 0.  
                * Time logs must be > 0 hours.  
                * Due date must be ≥ created date.  
                * Dependencies must point to valid task IDs.

                - **Persistence**  
                * Every CRUD operation saves directly to CSV.  
                * “Soft delete” sets `is_deleted=True` instead of removing rows.  
                * Reports always recompute from current CSVs on refresh.

                - **Concurrency**  
                * Multiple people editing the same CSV at the same time can conflict.  
                * For shared use, run TeamPulse on a single host and let teammates connect via LAN (`server_name="0.0.0.0"`).  
                * Alternatively, use the **Shared Inbox** feature in Admin to merge time logs from multiple machines.

                - **Extensibility**  
                * You can swap CSV persistence with SQLite/Postgres by replacing the load/save helpers.  
                * The Reports tab already isolates calculations into functions (`compute_utilization`, `burndown_series`, etc.), making it easy to extend.  
                * The UI is modular (each tab is self-contained), so you can add new tabs or metrics without touching the core.

                ---

                ✅ **Tip for Team Leads:** Run {APP_NAME} on one machine with `server_name="0.0.0.0"` so everyone shares the same board.  
                ✅ **Tip for Tech Users:** Back up the CSVs in `./data/` or version them in Git to keep history.  

                ---

                <div style="text-align:center; color:blue">
                    Created with the help of GPT5 by <b>Jyotendra</b>
                </div>
                """)




        return demo

# ----------------
# Entrypoint
# ----------------
def main():
    ui = build_ui()
    ui.launch()

if __name__ == "__main__":
    main()
