# Malay and Manglish fine-tune data

Builds training data for Malaysian English (Manglish, Malay–English
code-switching) and Bahasa Melayu, both formal and colloquial. The output plugs
straight into the trainer in [Domain Customization](../../../../docs/models/domain-customization.md#retraining):

```bash
python -m moonshine_voice.lora --train-manifest malaysia_data/train.jsonl ...
```

Rows in Chinese characters, Tamil or Jawi are dropped. The base model's English
tokenizer can't spell them. Romanised Malay and English are kept.

## Files

| File | What it is |
| --- | --- |
| `prepare_malaysia.py` | `inspect`, `build` and `check-tokenizer` commands |
| `malay_text.py` | Transcript cleanup shared by every source |
| `sources.json` | The candidate datasets. All start **disabled** |
| `test_prepare_malaysia.py` | Offline tests on synthetic audio |

## Candidate datasets

The licences below came from search results and haven't been verified. Read
each dataset card before you use it, and again before any commercial use.

| Source | Content | Licence (unverified) | Role |
| --- | --- | --- | --- |
| `mesolitica/Malaysian-STT-Whisper` | Large; Malaysian speech with Malay/English code-switching | not stated | train |
| `mesolitica/malaya-speech-malay-stt` | about 1.6M clips; machine-labelled Malay | not stated | train |
| IMDA National Speech Corpus | Thousands of hours of Singapore English (close to Manglish) | Singapore Open Data Licence | train |
| Common Voice Spontaneous `ms-MY` | about 6 h, colloquial | CC0 | train |
| Common Voice Scripted `ms` | about 3.6 h, formal, mostly unvalidated | CC0 | train |
| FLEURS `ms_my` | 9.5 h train + 2.3 h test, formal read speech | CC-BY-4.0 | train / test |
| MagicHub ASR-MalCSC | 5 h, conversational | CC BY-NC-ND (research only) | train |
| `Revolab/ASR-Benchmark-Public` | about 2.2 h of code-switching | CC-BY-4.0 annotations | **test only** |

There is no large, openly licensed Manglish speech corpus. For a commercial
product, IMDA, FLEURS and Common Voice are the safe base. Commercial
conversational Malay (MagicHub ASR-BigMalCSC, 372 h; Pangeanic, 72 h) has to be
licensed.

## Steps

```bash
pip install 'moonshine-voice[finetune]'
huggingface-cli login             # gated sets need you to accept terms on the Hub first
cd examples/python/finetune/malaysia
```

1. **Inspect each Hub source** and fix its `sources.json` entry (`files`,
   `revision`, `*_column`, `where`), then set `"enabled": true`:

   ```bash
   python prepare_malaysia.py inspect --repo mesolitica/Malaysian-STT-Whisper
   python prepare_malaysia.py inspect --repo google/fleurs \
       --revision refs/convert/parquet --files 'ms_my/test/*.parquet'
   ```

   Use `where` to keep only rows whose column matches a value, for example
   `"where": {"language": ["ms", "en"]}`. If a set keeps its audio in zips or
   as file paths, download it, then set `audio_root` or write a JSONL manifest
   and use `"type": "manifest"`. Relative `path`/`audio_root` values are resolved
   from the folder you run the command in.

2. **Download the non-Hub sources** (Common Voice from Mozilla Data Collective,
   IMDA NSC from IMDA, MagicHub) into `downloads/` and point their entries at
   the files. `"type": "table"` reads a TSV/CSV that has a header row, such as
   Common Voice's `validated.tsv`.

3. **Try a small build** first, then the full one:

   ```bash
   python prepare_malaysia.py build --only fleurs_ms_train,fleurs_ms_test --out-dir malaysia_data
   python prepare_malaysia.py build --out-dir malaysia_data
   ```

   This writes `train.jsonl`, `test.jsonl`, `test_<source>.jsonl`,
   `stats.json` and 16 kHz mono WAVs under `audio/<source>/`. Re-running
   reuses WAVs already on disk. Speakers are held out whole (3% per train
   source by default). Rows with no speaker id stay in train, and the build
   warns about them.

4. **Check the tokenizer** can spell what you built:

   ```bash
   python prepare_malaysia.py check-tokenizer --manifest malaysia_data/train.jsonl
   ```

   It also reports tokens per second. Malay words split into more tokens than
   English. If the rate is high, raise the `max_tokens_per_second` transcriber
   option above its 6.5 default for the adapted model, so long sentences aren't
   cut off as hallucinations.

5. **Train.** Manglish leans on English the model already knows. Bahasa Melayu
   is a new language for the English base model, so it needs the full fine-tune
   and many hours:

   ```bash
   cd ../../../..   # back to the repo root, or pass absolute paths
   python -m moonshine_voice.lora \
       --train-manifest examples/python/finetune/malaysia/malaysia_data/train.jsonl \
       --eval-manifest  examples/python/finetune/malaysia/malaysia_data/test.jsonl \
       --adapt full --train-hours 100 --eval --canary --output-dir ./ft_malaysia
   ```

   Start with `--sites both` (LoRA) on a few hours to check the pipeline end to
   end, then move to `--adapt full`. The English replay corpus stays on, which
   protects the English half of Manglish. Export and quantize as described in
   [Shipping the result](../../../../docs/models/domain-customization.md#shipping-the-result).

## Transcript rules

`build` applies the same pass to every source (`malay_text.normalize`):

- **Register is kept.** `tak`/`tidak`, `nak`/`hendak` and `aku`/`saya` are
  different words, and both are kept. Only written shorthand nobody says aloud
  (`yg`, `dgn`, `utk`, ...) is expanded. Turn that off with `--no-shorthand`, or
  add your own replacements with `--map extra.json`.
- **`--style plain`** (the default) lowercases text and strips punctuation,
  because these sources disagree on both. It keeps hyphens inside words
  (`kanak-kanak`). Use `--style cased` only if every source you enable is cased
  and punctuated.
- **`--particles join`** attaches a detached `lah` to the word before it
  (`okay lah` → `okaylah`, the standard Malay spelling). The default `keep`
  leaves `lah` as each corpus wrote it.
- **Noise tags** (`[LAUGHTER]`, `<unk>`, `{noise}`) are removed.
- **Numbers** aren't converted. `stats.json` reports the share of rows with
  digits in each source. If some sources use digits and others spell numbers
  out, choose one style.

## Tests

```bash
PYTHONPATH=language-bindings/python/src pytest examples/python/finetune/malaysia
```
