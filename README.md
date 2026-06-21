# EE200 Course Project — Q2 & Q3 Files

## Files
- `fingerprint.py` — core audio-fingerprinting functions (spectrogram, peak picking, hashing, database, matching). Imported by both the notebook (Q3A) and `app.py` (Q3B). Keep this file alongside both.
- `app.py` — Streamlit app for Q3B (single-clip mode + batch mode).
- `requirements.txt` — Python packages needed.

## IMPORTANT: swap in your real data before submitting
 To use your real data, no code changes are needed:put the provided song library as `.wav` files inside
   `data/songs/`. The notebook and the Streamlit app both
   auto-detect this folder.

If songs aren't `.wav` (e.g. `.mp3`), convert them first, e.g.
with `ffmpeg -i song.mp3 song.wav`, or extend `fingerprint.load_wav` /
add a librosa-based loader if you have internet access to install librosa.

## Running the Streamlit app locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploying (for the Q3B submission link)
1. Push this folder (including `data/songs/` with the real songs) to a
   public GitHub repo.
2. Go to https://share.streamlit.io, connect the repo, deploy `app.py`.