import json
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

DB_PATH   = Path.home() / ".transcribe" / "archive.db"
AUDIO_DIR = Path.home() / ".transcribe" / "audio"


async def init_db() -> None:
    DB_PATH.parent.mkdir(exist_ok=True)
    AUDIO_DIR.mkdir(exist_ok=True)
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
        await db.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                id       INTEGER PRIMARY KEY CHECK (id = 1),
                provider TEXT NOT NULL DEFAULT 'lmstudio',
                base_url TEXT NOT NULL DEFAULT 'http://localhost:1234',
                model    TEXT NOT NULL DEFAULT '',
                api_key  TEXT NOT NULL DEFAULT ''
            )
        """)
        try:
            await db.execute("ALTER TABLE archive ADD COLUMN speaker_names TEXT NOT NULL DEFAULT '{}'")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE archive ADD COLUMN summary TEXT")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE archive ADD COLUMN audio_ext TEXT")
        except Exception:
            pass
        await db.commit()


async def save_transcription(job_id: str, filename: str, segments: list, audio_ext: str | None = None) -> None:
    duration = segments[-1]["end"] if segments else 0
    speakers = len({s["speaker"] for s in segments})
    created_at = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            """INSERT INTO archive (id, filename, created_at, duration_s, speaker_count, result_json, speaker_names, audio_ext)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (job_id, filename, created_at, duration, speakers, json.dumps(segments), "{}", audio_ext),
        )
        await db.commit()


async def list_archive() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """SELECT id, filename, created_at, duration_s, speaker_count,
                      CASE WHEN summary IS NOT NULL AND summary != '' THEN 1 ELSE 0 END AS has_summary
               FROM archive ORDER BY created_at DESC"""
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


async def get_settings() -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT provider, base_url, model, api_key FROM settings WHERE id = 1") as cur:
            row = await cur.fetchone()
    if row is None:
        return {"provider": "lmstudio", "base_url": "http://localhost:1234", "model": "", "api_key": ""}
    return dict(row)


async def save_settings(provider: str, base_url: str, model: str, api_key: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO settings (id, provider, base_url, model, api_key) VALUES (1, ?, ?, ?, ?)",
            (provider, base_url, model, api_key),
        )
        await db.commit()


async def save_summary(entry_id: str, summary: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "UPDATE archive SET summary = ? WHERE id = ?", (summary, entry_id)
        )
        await db.commit()
        return cur.rowcount > 0


async def delete_archive_entry(entry_id: str) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT audio_ext FROM archive WHERE id = ?", (entry_id,)) as cur:
            row = await cur.fetchone()
        cur = await db.execute("DELETE FROM archive WHERE id = ?", (entry_id,))
        await db.commit()
        if cur.rowcount > 0 and row and row[0]:
            audio_path = AUDIO_DIR / f"{entry_id}{row[0]}"
            try:
                audio_path.unlink()
            except OSError:
                pass
        return cur.rowcount > 0
