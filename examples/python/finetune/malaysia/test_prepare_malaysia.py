"""Offline tests for the Malaysia data prep. Synthetic audio, no network.

    PYTHONPATH=language-bindings/python/src pytest examples/python/finetune/malaysia
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prepare_malaysia as pm  # noqa: E402
from malay_text import is_latin_text, normalize  # noqa: E402


def wav_bytes(seconds: float, rate: int = 44100, channels: int = 1) -> bytes:
    n = int(seconds * rate)
    data = 0.1 * np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, n)).astype(np.float32)
    if channels == 2:
        data = np.stack([data, data], axis=1)
    buffer = io.BytesIO()
    sf.write(buffer, data, rate, format="WAV")
    return buffer.getvalue()


# ---------------------------------------------------------------- text


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("Saya  nak pergi KL, okay lah!", "saya nak pergi kl okay lah"),
        ("Dia tak datang [LAUGHTER] semalam <unk>", "dia tak datang semalam"),
        ("Kanak-kanak tu main kat luar.", "kanak-kanak tu main kat luar"),
        ("yg ni utk awak dgn dia", "yang ni untuk awak dengan dia"),
        ("Don’t worry - it's fine", "don't worry it's fine"),
        ("[NOISE] <sil>", ""),
    ],
)
def test_normalize_plain(raw, expected):
    assert normalize(raw) == expected


def test_normalize_keeps_register():
    # Street and formal forms are different words, never merged.
    assert normalize("aku tak nak") == "aku tak nak"
    assert normalize("saya tidak hendak") == "saya tidak hendak"


def test_normalize_cased_keeps_punctuation_and_shorthand_case():
    assert normalize("Yg  ini, OK?", style="cased") == "Yang ini, OK?"


def test_join_particles_does_not_split_words():
    assert normalize("okay lah, jom", particles="join") == "okaylah jom"
    assert normalize("sekolah masalah", particles="join") == "sekolah masalah"


def test_extra_map_and_no_shorthand():
    assert normalize("yg x", expand_shorthand=False, extra_map={"x": "tak"}) == "yg tak"


def test_is_latin_text():
    assert is_latin_text("saya okay lah café")
    assert not is_latin_text("saya 我 okay")
    assert not is_latin_text("வணக்கம் lah")
    assert not is_latin_text("جاوي")  # Jawi


# ---------------------------------------------------------------- split


def test_speaker_split_is_disjoint_and_deterministic():
    rows = [
        {"speaker": f"s:{i % 40}", "seconds": 1.0, "audio": f"{i}.wav", "text": "x"}
        for i in range(400)
    ]
    train, test, unsplit = pm.split_rows(rows, "train", 0.2, seed=0)
    assert unsplit == 0 and train and test
    assert {r["speaker"] for r in train}.isdisjoint({r["speaker"] for r in test})
    again = pm.split_rows(rows, "train", 0.2, seed=0)
    assert [r["audio"] for r in again[1]] == [r["audio"] for r in test]


def test_rows_without_speaker_stay_in_train_and_test_role_goes_to_test():
    rows = [{"speaker": None, "seconds": 1.0}] * 3
    assert pm.split_rows(rows, "train", 0.5, 0) == (rows, [], 3)
    assert pm.split_rows(rows, "test", 0.5, 0) == ([], rows, 0)


# ---------------------------------------------------------------- process


def test_process_source_filters_resamples_and_resumes(tmp_path):
    records = [
        pm.Record("s", 1, "Saya OK lah", wav_bytes(1.0, channels=2), speaker="a", utt_id="clip/1.wav"),
        pm.Record("s", 2, "我们 lah", wav_bytes(1.0)),
        pm.Record("s", 3, "[NOISE]", wav_bytes(1.0)),
        pm.Record("s", 4, "pendek", wav_bytes(0.1)),
        pm.Record("s", 5, "panjang", wav_bytes(31.0, rate=16000)),
        pm.Record("s", 6, "rosak", b"not audio"),
        pm.Record("s", 7, "tapis", utt_id="__filtered__"),
        pm.Record("s", 8, "ada 3 orang", wav_bytes(2.0, rate=8000), speaker="b"),
    ]
    src = {"name": "demo"}
    opts = pm.TextOptions()
    rows, stats = pm.process_source(src, tmp_path, opts, 0.5, 30.0, iter(records))
    assert [r["text"] for r in rows] == ["saya ok lah", "ada 3 orang"]
    assert stats.dropped == {
        "non_latin_script": 1,
        "empty_text": 1,
        "too_short": 1,
        "too_long": 1,
        "audio_decode_error": 1,
        "filtered_by_where": 1,
    }
    assert stats.digit_rows == 1
    info = sf.info(str(tmp_path / rows[0]["audio"]))
    assert info.samplerate == 16000 and info.channels == 1
    assert rows[0]["speaker"] == "demo:a"
    assert rows[0]["audio"].startswith("audio/demo/1_")
    assert abs(rows[1]["seconds"] - 2.0) < 0.01

    # Second run reuses the WAVs and gives the same rows.
    again, _ = pm.process_source(src, tmp_path, opts, 0.5, 30.0, iter(records))
    assert again == rows


def test_hours_cap(tmp_path):
    records = (pm.Record("s", i, f"ayat {i}", wav_bytes(20.0, rate=16000)) for i in range(100))
    rows, stats = pm.process_source({"name": "cap", "hours": 60 / 3600}, tmp_path,
                                    pm.TextOptions(), 0.5, 30.0, records)
    assert len(rows) == 3 and stats.seconds == pytest.approx(60.0)


# ---------------------------------------------------------------- hub reader


class FakeHubFS:
    """Stands in for HfFileSystem: maps datasets/<repo>/... onto a local dir."""

    def __init__(self, root: Path, repo: str):
        self.root, self.prefix = root, f"datasets/{repo}"

    def glob(self, pattern):
        rel = pattern[len(self.prefix) + 1 :]
        return [f"{self.prefix}/{p.relative_to(self.root)}" for p in self.root.glob(rel)]

    def open(self, path, mode="rb"):
        return open(self.root / path[len(self.prefix) + 1 :], mode)


def test_hub_reader_detects_columns_and_applies_where(tmp_path):
    shard_dir = tmp_path / "data"
    shard_dir.mkdir()
    table = pa.table(
        {
            "audio": [{"bytes": wav_bytes(1.0), "path": "a.wav"},
                      {"bytes": wav_bytes(1.0), "path": "b.wav"}],
            "transcription": ["Hello lah", "skip me"],
            "speaker_id": ["s1", "s2"],
            "language": ["en", "zh"],
        }
    )
    pq.write_table(table, shard_dir / "train-0.parquet")
    src = {"name": "hub", "repo": "org/set", "where": {"language": ["en", "ms"]}}
    records = list(pm.iter_hub_records(src, fs=FakeHubFS(tmp_path, "org/set")))
    assert records[0].text == "Hello lah" and records[0].speaker == "s1"
    assert records[0].audio_bytes and records[1].utt_id == "__filtered__"


# ---------------------------------------------------------------- end to end


def test_build_end_to_end_with_table_and_manifest(tmp_path):
    clips = tmp_path / "clips"
    clips.mkdir()
    lines = ["client_id\tpath\tsentence"]
    for i in range(60):
        (clips / f"c{i}.wav").write_bytes(wav_bytes(1.0, rate=48000))
        lines.append(f"spk{i % 20}\tc{i}.wav\tAyat nombor {i} lah.")
    (tmp_path / "validated.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    test_dir = tmp_path / "test_audio"
    test_dir.mkdir()
    (test_dir / "t.wav").write_bytes(wav_bytes(1.5))
    (tmp_path / "test.jsonl").write_text(
        json.dumps({"audio": "test_audio/t.wav", "text": "Uji kod-suis, okay?"}) + "\n",
        encoding="utf-8",
    )
    config = {
        "sources": [
            {"name": "cv", "type": "table", "path": str(tmp_path / "validated.tsv"),
             "audio_root": str(clips), "role": "train"},
            {"name": "bench", "type": "manifest", "path": str(tmp_path / "test.jsonl"),
             "role": "test"},
            {"name": "off", "type": "hub", "repo": "x/y", "enabled": False},
        ]
    }
    (tmp_path / "sources.json").write_text(json.dumps(config), encoding="utf-8")
    out = tmp_path / "out"
    code = pm.main(["build", "--config", str(tmp_path / "sources.json"),
                    "--out-dir", str(out), "--test-fraction", "0.25"])
    assert code == 0

    train = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    test = [json.loads(l) for l in (out / "test.jsonl").read_text().splitlines()]
    assert len(train) + len(test) == 61
    assert any(r["source"] == "bench" and r["text"] == "uji kod-suis okay" for r in test)
    assert {r["speaker"] for r in train}.isdisjoint({r["speaker"] for r in test})
    stats = json.loads((out / "stats.json").read_text())
    assert [s["source"] for s in stats] == ["cv", "bench"]

    # The trainer's own loader reads the result.
    from moonshine_voice.lora.manifest import load_manifest

    loaded = load_manifest(str(out / "train.jsonl"))
    assert len(loaded) == len(train)
    assert Path(loaded[0].audio).is_file() and loaded[0].speaker.startswith("cv:")
