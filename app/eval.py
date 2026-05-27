"""Reference-free transcript quality evaluation.

There is no ground-truth transcript to diff against, so this does NOT judge
fine word accuracy ("Köln" vs "Cologne"). Instead it measures *failure
signatures* — things that are wrong by construction regardless of what was
actually said — so we can (a) prove the catastrophic failures (hallucination
loops, phantom speakers) are gone and (b) catch regressions when the pipeline
changes later.

Usage:
    python -m app.eval                 # score every archived transcript
    python -m app.eval --vad           # also measure VAD coverage (needs audio)
    python -m app.eval --baseline f.json   # save baseline, or compare & flag regressions
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import zlib
from collections import defaultdict
from pathlib import Path

# Well-documented Whisper hallucination artifacts (German + English). A couple
# of these in a real transcript are fine; a pile of them signals silence loops.
_FILLER_PHRASES = (
    "thank you", "thanks for watching", "please subscribe", "you're welcome",
    "let's go", "come on",
    "vielen dank", "untertitel", "untertitelung", "amara.org",
)

# Verdict thresholds. Heuristics, deliberately conservative so real content
# doesn't trip them.
_MAX_REPEAT_RUN = 4        # 4+ identical consecutive segments == loop
_COMPRESSION_LIMIT = 2.4   # Whisper's own repetitive-output threshold
# Fraction of segments that are NOTHING BUT filler. Counting filler *occurrences*
# over-flags German ("vielen dank" is legitimately said); a whole segment that
# is only filler is the hallucination signature.
_FILLER_SEGMENT_RATIO = 0.15
_THIN_SPEAKER_SHARE = 0.05 # speaker with < 5% of talk time == phantom cluster
_WORDS_OUTSIDE_VAD = 0.05  # > 5% of word-time over silence == hallucination


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower()).strip(".,!?…")


def _is_pure_filler(norm_text: str) -> bool:
    """True if the segment is nothing but filler phrases (a hallucination), as
    opposed to filler used legitimately inside a real sentence."""
    if not norm_text:
        return False
    stripped = norm_text
    for phrase in _FILLER_PHRASES:
        stripped = stripped.replace(phrase, " ")
    return not re.sub(r"[\s.,!?…]+", "", stripped)


def _compression_ratio(text: str) -> float:
    """Higher == more repetitive. Mirrors Whisper's internal anti-loop signal."""
    data = text.encode("utf-8")
    if len(data) < 16:
        return 1.0
    return len(data) / max(1, len(zlib.compress(data, 9)))


def _word_duration(seg: dict) -> float:
    words = seg.get("words", [])
    if words:
        return sum(max(0.0, w["end"] - w["start"]) for w in words)
    return max(0.0, seg.get("end", 0.0) - seg.get("start", 0.0))


def _vad_metrics(words: list[dict], regions: list[tuple[float, float]]) -> dict:
    if not words:
        return {"words_outside_vad": 0.0, "speech_covered": 0.0}

    def covers(t: float) -> bool:
        return any(s <= t <= e for s, e in regions)

    total = outside = 0.0
    for w in words:
        d = max(0.0, w["end"] - w["start"])
        total += d
        if not covers((w["start"] + w["end"]) / 2):
            outside += d

    speech_total = sum(e - s for s, e in regions) or 1.0
    covered = sum(
        (e - s)
        for s, e in regions
        if any(s <= (w["start"] + w["end"]) / 2 <= e for w in words)
    )
    return {
        # high == words transcribed where there's no speech (hallucination)
        "words_outside_vad": round(outside / max(total, 1e-9), 3),
        # low == VAD found speech that produced no words (dropped/missed speech)
        "speech_covered": round(covered / speech_total, 3),
    }


def score_transcript(segments: list[dict], speech_regions: list[tuple[float, float]] | None = None) -> dict:
    """Compute failure-signature metrics for one transcript (pure function)."""
    norm = [_norm(s.get("text", "")) for s in segments]
    words = [w for s in segments for w in s.get("words", [])]

    # longest run of identical consecutive segments
    max_run = run = 0
    prev = None
    for t in norm:
        run = run + 1 if (t and t == prev) else 1
        prev = t
        max_run = max(max_run, run)

    ratios = [_compression_ratio(s.get("text", "")) for s in segments if len(s.get("text", "")) >= 30]
    full = " ".join(norm)
    pure_filler = sum(1 for t in norm if _is_pure_filler(t))

    talk: dict[str, float] = defaultdict(float)
    for s in segments:
        talk[s.get("speaker", "?")] += _word_duration(s)
    grand = sum(talk.values()) or 1.0
    shares = {spk: dur / grand for spk, dur in talk.items()}

    result = {
        "n_segments": len(segments),
        "n_words": len(words),
        "n_speakers": len(talk),
        "n_thin_speakers": sum(1 for sh in shares.values() if sh < _THIN_SPEAKER_SHARE),
        "max_repeat_run": max_run,
        "max_compression": round(max(ratios), 2) if ratios else 1.0,
        "filler_hits": sum(full.count(p) for p in _FILLER_PHRASES),
        "filler_segment_ratio": round(pure_filler / max(len(norm), 1), 3),
        "speaker_shares": {k: round(v, 3) for k, v in sorted(shares.items(), key=lambda x: -x[1])},
    }
    if speech_regions is not None:
        result.update(_vad_metrics(words, speech_regions))
    return result


def flags(score: dict) -> list[str]:
    """Human-readable failure flags for a score; empty list == healthy."""
    out = []
    if score["max_repeat_run"] >= _MAX_REPEAT_RUN:
        out.append("repeat-loop")
    if score["max_compression"] >= _COMPRESSION_LIMIT:
        out.append("repetitive")
    if score["filler_segment_ratio"] >= _FILLER_SEGMENT_RATIO:
        out.append("filler")
    if score["n_thin_speakers"] >= 1:
        out.append("phantom-speaker")
    if score.get("words_outside_vad", 0.0) > _WORDS_OUTSIDE_VAD:
        out.append("words-over-silence")
    return out


# ── CLI ────────────────────────────────────────────────────────────────────

async def _gather(use_vad: bool) -> list[dict]:
    from .db import AUDIO_DIR, get_archive_entry, list_archive

    rows = []
    for summary in await list_archive():
        entry = await get_archive_entry(summary["id"])
        if not entry:
            continue
        regions = None
        if use_vad and entry.get("audio_ext"):
            audio = AUDIO_DIR / f"{entry['id']}{entry['audio_ext']}"
            if audio.exists():
                from .vad import detect_speech
                # VAD wants 16k mono; detect_speech reads via the model's loader.
                regions = detect_speech(str(audio))
        score = score_transcript(entry["segments"], regions)
        rows.append({"id": entry["id"], "filename": entry["filename"], "score": score, "flags": flags(score)})
    return rows


def _print_table(rows: list[dict]) -> None:
    hdr = f"{'id':8}  {'segs':>4} {'spk':>3} {'thin':>4} {'rep':>3} {'comp':>5} {'fill%':>5}  {'flags':24} filename"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        s = r["score"]
        print(
            f"{r['id'][:8]:8}  {s['n_segments']:>4} {s['n_speakers']:>3} {s['n_thin_speakers']:>4} "
            f"{s['max_repeat_run']:>3} {s['max_compression']:>5} {s['filler_segment_ratio']*100:>4.0f}%  "
            f"{(','.join(r['flags']) or 'ok'):24} {r['filename']}"
        )
    bad = [r for r in rows if r["flags"]]
    print(f"\n{len(rows)} transcripts | {len(bad)} flagged")


def _compare_baseline(rows: list[dict], path: Path) -> None:
    current = {r["id"]: r["score"] for r in rows}
    if not path.exists():
        path.write_text(json.dumps(current, indent=2))
        print(f"\nBaseline saved to {path} ({len(current)} entries).")
        return
    base = json.loads(path.read_text())
    regressions = []
    for r in rows:
        b = base.get(r["id"])
        if not b:
            continue
        s = r["score"]
        # a regression == a failure signal got worse vs baseline
        if s["max_repeat_run"] > b["max_repeat_run"] \
           or s["filler_segment_ratio"] > b.get("filler_segment_ratio", 0) + 0.02 \
           or s["n_thin_speakers"] > b["n_thin_speakers"] or s["max_compression"] > b["max_compression"] + 0.1 \
           or s.get("words_outside_vad", 0) > b.get("words_outside_vad", 0) + 0.02:
            regressions.append((r, b))
    if regressions:
        print(f"\n⚠️  {len(regressions)} REGRESSION(S) vs baseline:")
        for r, b in regressions:
            print(f"  {r['id'][:8]} {r['filename']}: now {r['score']} | was {b}")
    else:
        print(f"\n✓ no regressions vs baseline ({path}).")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reference-free transcript quality eval.")
    ap.add_argument("--vad", action="store_true", help="also measure VAD coverage (needs audio, slower)")
    ap.add_argument("--baseline", type=Path, help="save baseline if missing, else compare & flag regressions")
    args = ap.parse_args()

    rows = asyncio.run(_gather(args.vad))
    rows.sort(key=lambda r: (not r["flags"], r["filename"]))  # flagged first
    _print_table(rows)
    if args.baseline:
        _compare_baseline(rows, args.baseline)


if __name__ == "__main__":
    main()
