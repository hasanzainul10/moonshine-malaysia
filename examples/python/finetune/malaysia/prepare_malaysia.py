#!/usr/bin/env python3
"""Turn Malay and Manglish speech corpora into Moonshine fine-tune manifests.

Three commands:

    inspect          show a Hub dataset's shards, columns and a few rows, so you
                     can fill in its entry in sources.json
    build            download, clean, filter and split every enabled source into
                     train.jsonl / test.jsonl (16 kHz mono WAV next to them)
    check-tokenizer  confirm the model's tokenizer can spell the transcripts and
                     measure tokens per second of audio

The output is exactly what ``python -m moonshine_voice.lora --train-manifest``
reads. Needs ``pip install 'moonshine-voice[finetune]'``; no extra packages.
See README.md in this directory for the walkthrough.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence
from urllib.parse import quote

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from malay_text import has_digits, is_latin_text, normalize  # noqa: E402

SAMPLE_RATE = 16000
DEFAULT_MODEL = "moonshine-ai/moonshine-streaming-medium"

AUDIO_COLUMNS = ("audio", "audio_filename", "speech", "wav", "file")
TEXT_COLUMNS = (
    "text",
    "transcription",
    "sentence",
    "transcript",
    "raw_transcription",
    "normalized_text",
)
SPEAKER_COLUMNS = ("speaker_id", "speaker", "client_id", "spk_id", "spk", "reader_id")
ID_COLUMNS = ("id", "utterance_id", "audio_id", "key", "segment_id")


# ---------------------------------------------------------------- records


@dataclass
class Record:
    """One clip before cleaning. ``audio`` is raw bytes or a file path."""

    shard: str
    row: int
    text: str
    audio_bytes: Optional[bytes] = None
    audio_path: Optional[str] = None
    speaker: Optional[str] = None
    utt_id: Optional[str] = None


@dataclass
class SourceStats:
    seen: int = 0
    kept: int = 0
    seconds: float = 0.0
    digit_rows: int = 0
    dropped: Counter = field(default_factory=Counter)
    speakers: set = field(default_factory=set)


def pick_column(names: Sequence[str], candidates: Sequence[str], override: Optional[str]):
    if override:
        if override not in names:
            raise KeyError(f"column {override!r} not in {list(names)}")
        return override
    lowered = {n.lower(): n for n in names}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    return None


def _passes(row: dict, where: Dict[str, list]) -> bool:
    for column, allowed in where.items():
        if row.get(column) not in allowed:
            return False
    return True


def _audio_fields(value):
    """HF parquet stores audio as {bytes, path}; some sets store just a path."""
    if isinstance(value, dict):
        return value.get("bytes"), value.get("path")
    if isinstance(value, (bytes, bytearray)):
        return bytes(value), None
    if isinstance(value, str):
        return None, value
    return None, None


# ---------------------------------------------------------------- Hub sources


def _hub_root(repo: str, revision: Optional[str]) -> str:
    root = f"datasets/{repo}"
    if revision:
        root += "@" + quote(revision, safe="")
    return root


def hub_shards(fs, repo: str, files: str, revision: Optional[str]) -> List[str]:
    return sorted(fs.glob(f"{_hub_root(repo, revision)}/{files}"))


def iter_hub_records(src: dict, fs=None) -> Iterator[Record]:
    import pyarrow.parquet as pq

    if fs is None:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
    shards = hub_shards(fs, src["repo"], src.get("files", "**/*.parquet"), src.get("revision"))
    if not shards:
        raise FileNotFoundError(
            f"{src['name']}: no files match {src.get('files', '**/*.parquet')!r} in "
            f"{src['repo']}. Run `prepare_malaysia.py inspect --repo {src['repo']}` "
            "(try --revision refs/convert/parquet for script-based datasets)."
        )
    where = src.get("where", {})
    for shard in shards:
        with fs.open(shard, "rb") as handle:
            parquet = pq.ParquetFile(handle)
            names = parquet.schema_arrow.names
            audio_col = pick_column(names, AUDIO_COLUMNS, src.get("audio_column"))
            text_col = pick_column(names, TEXT_COLUMNS, src.get("text_column"))
            spk_col = pick_column(names, SPEAKER_COLUMNS, src.get("speaker_column"))
            id_col = pick_column(names, ID_COLUMNS, src.get("id_column"))
            if audio_col is None or text_col is None:
                raise KeyError(
                    f"{src['name']}: cannot find audio/text columns in {names}; "
                    "set audio_column / text_column in sources.json"
                )
            wanted = {audio_col, text_col, *where}
            wanted |= {c for c in (spk_col, id_col) if c}
            row_no = 0
            for batch in parquet.iter_batches(batch_size=32, columns=sorted(wanted)):
                for row in batch.to_pylist():
                    row_no += 1
                    if not _passes(row, where):
                        yield Record(shard, row_no, "", utt_id="__filtered__")
                        continue
                    blob, path = _audio_fields(row.get(audio_col))
                    yield Record(
                        shard=shard,
                        row=row_no,
                        text=str(row.get(text_col) or ""),
                        audio_bytes=blob,
                        audio_path=path,
                        speaker=None if not spk_col or row.get(spk_col) is None
                        else str(row.get(spk_col)),
                        utt_id=None if not id_col else str(row.get(id_col)),
                    )


def iter_manifest_records(src: dict) -> Iterator[Record]:
    """A local JSONL/JSON/TSV manifest: Common Voice, IMDA, MagicHub downloads."""
    from moonshine_voice.lora.manifest import load_manifest

    rows = load_manifest(src["path"], src.get("data_root"))
    for i, row in enumerate(rows, 1):
        speaker = row.speaker
        key = src.get("speaker_column")
        if key and key in row.extra:
            speaker = str(row.extra[key])
        yield Record(
            shard=src["path"],
            row=i,
            text=row.text,
            audio_path=row.audio,
            speaker=speaker,
            utt_id=row.utterance_id,
        )


def iter_table_records(src: dict) -> Iterator[Record]:
    """A TSV/CSV with a header row, e.g. Common Voice's validated.tsv.

    Paths are resolved against ``audio_root`` (Common Voice: the clips/ folder).
    """
    path = Path(src["path"])
    delimiter = "," if path.suffix.lower() == ".csv" else "\t"
    where = src.get("where", {})
    with path.open(encoding="utf-8", newline="") as handle:
        quoting = csv.QUOTE_MINIMAL if delimiter == "," else csv.QUOTE_NONE
        reader = csv.DictReader(handle, delimiter=delimiter, quoting=quoting)
        names = reader.fieldnames or []
        audio_col = pick_column(names, ("path",) + AUDIO_COLUMNS, src.get("audio_column"))
        text_col = pick_column(names, TEXT_COLUMNS, src.get("text_column"))
        spk_col = pick_column(names, SPEAKER_COLUMNS, src.get("speaker_column"))
        id_col = pick_column(names, ID_COLUMNS, src.get("id_column"))
        if audio_col is None or text_col is None:
            raise KeyError(f"{src['name']}: cannot find audio/text columns in {names}")
        for i, row in enumerate(reader, 1):
            if not _passes(row, where):
                yield Record(str(path), i, "", utt_id="__filtered__")
                continue
            yield Record(
                shard=str(path),
                row=i,
                text=row.get(text_col) or "",
                audio_path=row.get(audio_col),
                speaker=(row.get(spk_col) or None) if spk_col else None,
                utt_id=(row.get(id_col) if id_col else None) or row.get(audio_col),
            )


def iter_records(src: dict) -> Iterator[Record]:
    kind = src.get("type", "hub")
    if kind == "hub":
        return iter_hub_records(src)
    if kind == "manifest":
        return iter_manifest_records(src)
    if kind == "table":
        return iter_table_records(src)
    raise ValueError(f"{src.get('name')}: unknown source type {kind!r}")


# ---------------------------------------------------------------- audio


def to_mono_16k(data, rate):
    import numpy as np

    if data.ndim > 1:
        data = data.mean(axis=1)
    if int(rate) != SAMPLE_RATE:
        from scipy.signal import resample_poly

        gcd = math.gcd(int(rate), SAMPLE_RATE)
        data = resample_poly(data, SAMPLE_RATE // gcd, int(rate) // gcd)
    return np.asarray(data, dtype=np.float32)


def decode_audio(record: Record, audio_root: Optional[str]):
    import soundfile as sf

    if record.audio_bytes:
        data, rate = sf.read(io.BytesIO(record.audio_bytes), dtype="float32")
    elif record.audio_path:
        path = Path(record.audio_path)
        if not path.is_absolute() and audio_root:
            path = Path(audio_root) / path
        if not path.is_file():
            raise FileNotFoundError(str(path))
        data, rate = sf.read(str(path), dtype="float32")
    else:
        raise ValueError("record has no audio")
    return to_mono_16k(data, rate)


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def clip_name(record: Record) -> str:
    base = _UNSAFE.sub("_", Path(record.utt_id).stem if record.utt_id else "")[:80].strip("_")
    digest = hashlib.sha1(f"{record.shard}:{record.row}".encode()).hexdigest()[:10]
    return f"{base}_{digest}" if base else digest


# ---------------------------------------------------------------- build


@dataclass
class TextOptions:
    style: str = "plain"
    particles: str = "keep"
    expand_shorthand: bool = True
    extra_map: Optional[dict] = None

    def apply(self, text: str) -> str:
        return normalize(
            text,
            style=self.style,
            particles=self.particles,
            expand_shorthand=self.expand_shorthand,
            extra_map=self.extra_map,
        )


def process_source(
    src: dict,
    out_dir: Path,
    text_opts: TextOptions,
    min_seconds: float,
    max_seconds: float,
    records: Optional[Iterator[Record]] = None,
) -> tuple:
    """Clean, decode and write one source. Returns (rows, SourceStats)."""
    import soundfile as sf

    name = src["name"]
    cap = src.get("hours")
    stats = SourceStats()
    rows: List[dict] = []
    clip_dir = out_dir / "audio" / name
    clip_dir.mkdir(parents=True, exist_ok=True)
    records = records if records is not None else iter_records(src)
    for record in records:
        if cap is not None and stats.seconds >= cap * 3600:
            break
        stats.seen += 1
        if record.utt_id == "__filtered__":
            stats.dropped["filtered_by_where"] += 1
            continue
        text = text_opts.apply(record.text)
        if not text:
            stats.dropped["empty_text"] += 1
            continue
        if not is_latin_text(text):
            stats.dropped["non_latin_script"] += 1
            continue
        wav = clip_dir / f"{clip_name(record)}.wav"
        if wav.is_file():
            seconds = sf.info(str(wav)).duration
        else:
            try:
                wave = decode_audio(record, src.get("audio_root"))
            except FileNotFoundError:
                stats.dropped["audio_missing"] += 1
                continue
            except Exception:  # noqa: BLE001 - corrupt clips are common in scraped sets
                stats.dropped["audio_decode_error"] += 1
                continue
            seconds = len(wave) / SAMPLE_RATE
            if seconds < min_seconds:
                stats.dropped["too_short"] += 1
                continue
            if seconds > max_seconds:
                stats.dropped["too_long"] += 1
                continue
            sf.write(str(wav), wave, SAMPLE_RATE, subtype="PCM_16")
        if not (min_seconds <= seconds <= max_seconds):
            stats.dropped["too_long" if seconds > max_seconds else "too_short"] += 1
            continue
        speaker = f"{name}:{record.speaker}" if record.speaker else None
        stats.kept += 1
        stats.seconds += seconds
        stats.digit_rows += has_digits(text)
        if speaker:
            stats.speakers.add(speaker)
        rows.append(
            {
                "audio": str(wav.relative_to(out_dir)),
                "text": text,
                "seconds": round(seconds, 3),
                "speaker": speaker,
                "id": wav.stem,
                "source": name,
            }
        )
        if stats.kept % 1000 == 0:
            print(f"  {name}: {stats.kept} clips, {stats.seconds / 3600:.2f} h", flush=True)
    return rows, stats


def is_test_speaker(speaker: str, fraction: float, seed: int) -> bool:
    digest = hashlib.sha1(f"{seed}:{speaker}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF < fraction


def split_rows(rows: List[dict], role: str, test_fraction: float, seed: int):
    """Hold out whole speakers. Rows without a speaker can't be held out safely."""
    if role == "test":
        return [], rows, 0
    train, test, unsplittable = [], [], 0
    for row in rows:
        if row["speaker"] is None:
            train.append(row)
            unsplittable += 1
        elif is_test_speaker(row["speaker"], test_fraction, seed):
            test.append(row)
        else:
            train.append(row)
    return train, test, unsplittable


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_config(path: str) -> List[dict]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    sources = config["sources"] if isinstance(config, dict) else config
    names = [s["name"] for s in sources]
    if len(set(names)) != len(names):
        raise ValueError("source names must be unique")
    for src in sources:
        if src.get("role", "train") not in ("train", "test"):
            raise ValueError(f"{src['name']}: role must be train or test")
    return sources


def cmd_build(args) -> int:
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [s for s in load_config(args.config) if s.get("enabled", True)]
    if args.only:
        wanted = set(args.only.split(","))
        sources = [s for s in sources if s["name"] in wanted]
    if not sources:
        print("no enabled sources", file=sys.stderr)
        return 1
    extra_map = json.loads(Path(args.map).read_text(encoding="utf-8")) if args.map else None
    text_opts = TextOptions(args.style, args.particles, not args.no_shorthand, extra_map)

    train_all, test_all, report = [], [], []
    for src in sources:
        print(f"[{src['name']}] {src.get('license', 'licence: check the dataset card')}")
        rows, stats = process_source(src, out_dir, text_opts, args.min_seconds, args.max_seconds)
        train, test, unsplittable = split_rows(
            rows, src.get("role", "train"), args.test_fraction, args.seed
        )
        train_all += train
        test_all += test
        if test:
            write_jsonl(out_dir / f"test_{src['name']}.jsonl", test)
        report.append(
            {
                "source": src["name"],
                "role": src.get("role", "train"),
                "kept": stats.kept,
                "seen": stats.seen,
                "train_hours": round(sum(r["seconds"] for r in train) / 3600, 3),
                "test_hours": round(sum(r["seconds"] for r in test) / 3600, 3),
                "speakers": len(stats.speakers),
                "no_speaker_rows": unsplittable,
                "rows_with_digits_pct": round(100 * stats.digit_rows / max(1, stats.kept), 1),
                "dropped": dict(stats.dropped),
            }
        )

    # The trainer takes its dev slice from the tail of the train manifest, so
    # shuffle or the dev set is just the last source.
    random.Random(args.seed).shuffle(train_all)
    write_jsonl(out_dir / "train.jsonl", train_all)
    write_jsonl(out_dir / "test.jsonl", test_all)
    (out_dir / "stats.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_report(report, out_dir)
    return 0


def print_report(report: List[dict], out_dir: Path) -> None:
    print()
    print(f"{'source':24} {'role':5} {'train h':>8} {'test h':>7} {'spk':>6} {'digits%':>7}  dropped")
    for r in report:
        dropped = ", ".join(f"{k}={v}" for k, v in sorted(r["dropped"].items())) or "-"
        print(
            f"{r['source'][:24]:24} {r['role']:5} {r['train_hours']:8.2f} {r['test_hours']:7.2f} "
            f"{r['speakers']:6d} {r['rows_with_digits_pct']:7.1f}  {dropped}"
        )
    print()
    for r in report:
        if r["role"] == "train" and r["no_speaker_rows"]:
            print(
                f"warning: {r['source']}: {r['no_speaker_rows']} rows have no speaker id, so "
                "they all stay in train. Set speaker_column, or score on other sources."
            )
    digit = [r["source"] for r in report if r["rows_with_digits_pct"] > 5]
    if digit:
        print(
            "note: " + ", ".join(digit) + " write numbers as digits in over 5% of rows. "
            "If other sources spell them out, the model will learn both; pick one."
        )
    print(f"wrote {out_dir / 'train.jsonl'}, {out_dir / 'test.jsonl'}, {out_dir / 'stats.json'}")


# ---------------------------------------------------------------- inspect


def cmd_inspect(args) -> int:
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    shards = hub_shards(fs, args.repo, args.files, args.revision)
    print(f"{len(shards)} files match {args.files!r} in {args.repo}"
          + (f"@{args.revision}" if args.revision else ""))
    for shard in shards[:8]:
        print("  " + shard)
    if not shards:
        top = fs.ls(_hub_root(args.repo, args.revision), detail=False)
        print("top level: " + ", ".join(Path(p).name for p in top))
        print("no parquet found: try --revision refs/convert/parquet, a different --files "
              "glob, or download the set and use a manifest source")
        return 1
    with fs.open(shards[0], "rb") as handle:
        parquet = pq.ParquetFile(handle)
        print(f"\nschema of {Path(shards[0]).name} ({parquet.metadata.num_rows} rows):")
        print(parquet.schema_arrow)
        batch = next(parquet.iter_batches(batch_size=args.rows))
        for row in batch.to_pylist()[: args.rows]:
            shown = {}
            for k, v in row.items():
                if isinstance(v, (bytes, bytearray)):
                    shown[k] = f"<{len(v)} bytes>"
                elif isinstance(v, dict) and "bytes" in v:
                    blob = v.get("bytes")
                    shown[k] = {"bytes": len(blob) if blob else None, "path": v.get("path")}
                else:
                    shown[k] = v if not isinstance(v, str) or len(v) < 200 else v[:200] + "..."
            print(json.dumps(shown, ensure_ascii=False, default=str))
        names = parquet.schema_arrow.names
        print("\nauto-detected: audio=%s text=%s speaker=%s id=%s" % (
            pick_column(names, AUDIO_COLUMNS, None),
            pick_column(names, TEXT_COLUMNS, None),
            pick_column(names, SPEAKER_COLUMNS, None),
            pick_column(names, ID_COLUMNS, None),
        ))
    return 0


# ---------------------------------------------------------------- tokenizer


def cmd_check_tokenizer(args) -> int:
    from transformers import AutoProcessor

    tokenizer = AutoProcessor.from_pretrained(args.model).tokenizer
    words: Counter = Counter()
    seconds = 0.0
    total_tokens = 0
    with open(args.manifest, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            words.update(row["text"].split())
            seconds += float(row.get("seconds") or 0)
            total_tokens += len(tokenizer(row["text"], add_special_tokens=False)["input_ids"])

    unk = tokenizer.unk_token_id
    bad, per_word = [], []
    for word, count in words.items():
        ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        per_word.append(len(ids))
        decoded = tokenizer.decode(ids).strip()
        if decoded != word or (unk is not None and unk in ids):
            bad.append((count, word, decoded))
    bad.sort(reverse=True)
    occurrences = sum(words.values())
    bad_occ = sum(c for c, _, _ in bad)
    print(f"{len(words)} distinct words, {occurrences} total")
    print(f"tokens per word (distinct): {sum(per_word) / max(1, len(per_word)):.2f}")
    print(f"words that do not round-trip: {len(bad)} distinct, "
          f"{100 * bad_occ / max(1, occurrences):.3f}% of occurrences")
    for count, word, decoded in bad[: args.top]:
        print(f"  {count:7d}  {word!r} -> {decoded!r}")
    if seconds:
        rate = total_tokens / seconds
        print(f"\n{rate:.2f} tokens per second of audio (runtime cut-off default 6.5)")
        if rate * 1.5 > 6.5:
            suggested = math.ceil(rate * 2)
            print(f"set max_tokens_per_second to about {suggested} when loading the adapted "
                  "model, or long Malay words can be cut off as 'hallucination'")
    return 1 if bad_occ / max(1, occurrences) > 0.01 else 0


# ---------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("inspect", help="show a Hub dataset's files, columns and rows")
    p.add_argument("--repo", required=True)
    p.add_argument("--revision", default=None,
                   help="e.g. refs/convert/parquet for script-based datasets")
    p.add_argument("--files", default="**/*.parquet")
    p.add_argument("--rows", type=int, default=3)
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("build", help="write train.jsonl / test.jsonl")
    p.add_argument("--config", default=str(HERE / "sources.json"))
    p.add_argument("--out-dir", default="malaysia_data")
    p.add_argument("--only", default=None, help="comma-separated source names")
    p.add_argument("--style", choices=["plain", "cased"], default="plain",
                   help="plain lowercases and drops punctuation (default; sources "
                   "disagree on casing). cased keeps both")
    p.add_argument("--particles", choices=["keep", "join"], default="keep",
                   help="join attaches a detached 'lah' to the word before it")
    p.add_argument("--no-shorthand", action="store_true",
                   help="do not expand chat shorthand like yg/dgn/utk")
    p.add_argument("--map", default=None, help="JSON {word: replacement} applied after shorthand")
    p.add_argument("--test-fraction", type=float, default=0.03,
                   help="share of speakers held out of each train source")
    p.add_argument("--min-seconds", type=float, default=0.5)
    p.add_argument("--max-seconds", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("check-tokenizer", help="can the model spell these transcripts?")
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--top", type=int, default=30)
    p.set_defaults(func=cmd_check_tokenizer)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
