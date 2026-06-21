"""
app.py  --  Q3B: 'Zapptain America'
------------------------------------
A simple Streamlit app wrapping the Q3A audio-fingerprinting pipeline.

Run locally with:
    streamlit run app.py

Deploy for free on Streamlit Community Cloud:
    1. Push this folder (app.py, fingerprint.py, requirements.txt, and a
       data/songs/ folder with the provided song library as .wav files)
       to a public GitHub repo.
    2. Go to https://share.streamlit.io , connect the repo, and deploy.
    3. Make sure data/songs/ is committed to the repo so the database is
       available immediately when the app starts (per the assignment's
       "Song Database" requirement).

Two modes (as required by the assignment):
    - Single-clip mode : upload one query clip -> shows spectrogram,
      constellation map, offset histogram, and the predicted song.
    - Batch mode        : upload several query clips -> writes results.csv
      with exactly two columns: filename, prediction
"""

import os
import io
import pickle

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import fingerprint as fp

st.set_page_config(page_title="Song Identifier", layout="wide")

SR = 11025
SONG_FOLDER = "data/songs"
DB_CACHE_PATH = "song_database.pkl"


# ----------------------------------------------------------------------
# Database: build once, cache to disk so re-runs are instant
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Indexing song database...")
def get_database():
    song_files = fp.list_songs(SONG_FOLDER)

    if not song_files:
        # Fallback so the app is still demonstrable without the real
        # provided song library present (e.g. first deploy / local testing).
        st.warning(
            f"No .wav files found in '{SONG_FOLDER}/'. "
            "Using a small SYNTHETIC demo library instead -- "
            "add the real provided songs there for the actual submission."
        )
        songs_audio = {
            "song_alpha": fp.make_synthetic_song("song_alpha", sr=SR, seed=1),
            "song_beta":  fp.make_synthetic_song("song_beta",  sr=SR, seed=2),
            "song_gamma": fp.make_synthetic_song("song_gamma", sr=SR, seed=3),
        }
        songs_list = [(name, audio, SR) for name, audio in songs_audio.items()]
    else:
        songs_list = []
        for name, path in song_files:
            audio, sr = fp.load_wav(path, target_sr=SR)
            songs_list.append((name, audio, sr))

    db = fp.build_database(songs_list, nperseg=1024, fan_out=5, use_pairs=True)
    song_names = [s[0] for s in songs_list]
    return db, song_names


def load_query_audio(uploaded_file):
    """Read an uploaded .wav file into a normalized mono numpy array at SR."""
    from scipy.io import wavfile
    from scipy import signal as sps

    sr, data = wavfile.read(io.BytesIO(uploaded_file.read()))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    data = data / (np.max(np.abs(data)) + 1e-9)
    if sr != SR:
        n_new = int(len(data) * SR / sr)
        data = sps.resample(data, n_new)
    return data


def identify_clip(audio, db):
    """Run the full pipeline and return everything needed to render the
    visuals: predicted song, spectrogram, peaks, offset histogram."""
    f_spec, t_spec, Sxx = fp.spectrogram(audio, SR, nperseg=1024)
    peaks = fp.find_peaks_2d(Sxx)
    best_song, offset_hist, _ = fp.match_query(audio, SR, db, nperseg=1024, fan_out=5, use_pairs=True)
    return best_song, f_spec, t_spec, Sxx, peaks, offset_hist


def plot_spectrogram_and_constellation(f_spec, t_spec, Sxx, peaks):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.pcolormesh(t_spec, f_spec, 20 * np.log10(Sxx + 1e-9), shading="auto", cmap="gray_r")
    if peaks:
        freq_idx = [p[0] for p in peaks]
        time_idx = [p[1] for p in peaks]
        ax.scatter(t_spec[time_idx], f_spec[freq_idx], c="red", s=15)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Spectrogram + constellation (red dots = fingerprint peaks)")
    return fig


def plot_offset_histogram(offset_hist):
    fig, ax = plt.subplots(figsize=(8, 3))
    for name, hist in offset_hist.items():
        offsets = sorted(hist.keys())
        counts = [hist[o] for o in offsets]
        ax.plot(offsets, counts, marker="o", ms=3, label=name)
    ax.set_xlabel("time offset (spectrogram bins)")
    ax.set_ylabel("matching hash count")
    ax.set_title("Offset histogram (tall sharp peak = the matched song)")
    ax.legend()
    return fig


# ----------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------
st.title("Sonic Signatures - Song Identifier")
st.caption("EE200 Course Project")

db, song_names = get_database()
st.success(f"Database ready: {len(song_names)} songs indexed -> {', '.join(song_names)}")

mode = st.radio("Choose a mode:", ["Single-clip mode", "Batch mode"], horizontal=True)

if mode == "Single-clip mode":
    st.subheader("Upload a single query clip (.wav)")
    uploaded = st.file_uploader("Query clip", type=["wav"])

    if uploaded is not None:
        audio = load_query_audio(uploaded)
        best_song, f_spec, t_spec, Sxx, peaks, offset_hist = identify_clip(audio, db)

        st.markdown(f"### Predicted song: **{best_song if best_song else 'No match found'}**")

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(plot_spectrogram_and_constellation(f_spec, t_spec, Sxx, peaks))
        with col2:
            st.pyplot(plot_offset_histogram(offset_hist))

else:
    st.subheader("Upload multiple query clips (.wav) for batch identification")
    uploaded_files = st.file_uploader("Query clips", type=["wav"], accept_multiple_files=True)

    if uploaded_files:
        rows = []
        progress = st.progress(0.0)
        for i, uf in enumerate(uploaded_files):
            audio = load_query_audio(uf)
            best_song, *_ = identify_clip(audio, db)
            filename_no_ext = os.path.splitext(uf.name)[0]
            rows.append({"filename": uf.name, "prediction": best_song if best_song else ""})
            progress.progress((i + 1) / len(uploaded_files))

        results_df = pd.DataFrame(rows, columns=["filename", "prediction"])
        st.dataframe(results_df)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download results.csv", data=csv_bytes,
                            file_name="results.csv", mime="text/csv")
