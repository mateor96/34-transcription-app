import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

DB_PATH = Path.home() / ".transcribe" / "archive.db"


async def init_db() -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS archive (
                id            TEXT PRIMARY KEY,
                filename      TEXT NOT NULL,
                created_at    TEXT NOT NULL,
                duration_s    REAL,
                speaker_count INTEGER,
                result_json   TEXT NOT NULL,
                speaker_names TEXT NOT NULL DEFAULT '{}'
            )
        """)
        # Migrate existing DBs that don't have the column yet
        try:
            await db.execute("ALTER TABLE archive ADD COLUMN speaker_names TEXT NOT NULL DEFAULT '{}'")
        except Exception:
            pass
        await db.commit()


async def save_transcription(job_id: str, filename: str, segments: list) -> None:
    duration = segments[-1]["end"] if segments else 0
    speakers = len({s["speaker"] for s in segments})
    created_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO archive VALUES (?, ?, ?, ?, ?, ?, ?)",
            (job_id, filename, created_at, duration, speakers, json.dumps(segments), "{}"),
        )
        await db.commit()


async def list_archive() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT id, filename, created_at, duration_s, speaker_count FROM archive ORDER BY created_at DESC"
        ) as cur:
            return [dict(r) for r in await cur.fetchall()]


async def get_archive_entry(entry_id: str) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM archive WHERE id = ?", (entry_id,)) as cur:
            row = await cur.fetchone()
    if row is None:
        return None
    d = dict(row)
    d["segments"]      = json.loads(d.pop("result_json"))
    d["speaker_names"] = json.loads(d.get("speaker_names") or "{}")
    return d


async def update_speaker_names(entry_id: str, names: dict) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "UPDATE archive SET speaker_names = ? WHERE id = ?",
            (json.dumps(names), entry_id),
        )
        await db.commit()
        return cur.rowcount > 0


async def update_filename(entry_id: str, filename: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "UPDATE archive SET filename = ? WHERE id = ?",
            (filename, entry_id),
        )
        await db.commit()
        return cur.rowcount > 0


async def delete_archive_entry(entry_id: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("DELETE FROM archive WHERE id = ?", (entry_id,))
        await db.commit()
        return cur.rowcount > 0
