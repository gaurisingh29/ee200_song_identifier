"""
app.py
----------------
High-efficiency Streamlit Dashboard using native frontend charts 
optimized for static, square elements with raw point visualization.
"""

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import librosa
from PIL import Image, ImageDraw

import fingerprint as fp

st.set_page_config(
    page_title="Sonic Signatures",
    page_icon=":material/graphic_eq:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --------------------------------------------------------------------------
# Custom theme / styling
# --------------------------------------------------------------------------
st.markdown("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Sora:wght=400;600;700;800&family=Inter:wght=400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Page background */
    .stApp {
        background: radial-gradient(circle at 10% 0%, #1b1530 0%, #0e0b1a 45%, #090713 100%);
    }

    /* Hide default Streamlit chrome */
    header[data-testid="stHeader"] {
        background: transparent;
    }

    /* Hero banner using a local file named hero.jpg */
    .hero {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        padding: 3.5rem 2.4rem;
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(14, 11, 26, 0.6) 0%, rgba(9, 7, 19, 0.85) 100%), url('app/static/hero.jpg'), url('hero.jpg');
        background-size: cover;
        background-position: center;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 18px 45px -20px rgba(124,58,237,0.4);
        margin-bottom: 2rem;
    }
    .hero-title {
        font-family: 'Helvetica', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #f5f3ff 0%, #c4b5fd 60%, #7dd3fc 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        margin: 0;
        text-align: center;
    }
    .hero-subtitle {
        font-family: 'Courier New', Courier, monospace;
        color: #b7b2cc;
        font-size: 1.05rem;
        margin: 0.6rem 0 0 0;
        max-width: 680px;
        line-height: 1.5;
        text-align: center;
    }
    .hero-pill-row {
        margin-top: 1.4rem;
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        justify-content: center;
    }
    .hero-pill {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        font-weight: 600;
        color: #ddd6fe;
        background: rgba(14, 11, 26, 0.7);
        border: 1px solid rgba(255,255,255,0.15);
        padding: 0.35rem 1rem;
        border-radius: 999px;
        letter-spacing: 0.01em;
        backdrop-filter: blur(4px);
    }

    /* Section card wrapper (legacy class kept for compatibility, real
       styling now applied via div[class*="st-key-panelcard"] below) */

    .library-track-title {
        font-family: 'Helvetica', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        color: #f4f2ff;
        margin-bottom: 0.2rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        text-align: center;
    }
    .library-track-meta {
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.82rem;
        color: #9893ad;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .library-track-meta strong {
        color: #a78bfa;
    }

    .section-heading {
        font-family: 'Helvetica', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 700;
        font-size: 1.18rem;
        color: #f4f2ff;
        margin-bottom: 0.15rem;
        text-align: center;
    }
    .section-caption {
        font-family: 'Courier New', Courier, monospace;
        color: #9893ad;
        font-size: 0.9rem;
        margin-bottom: 1.05rem;
        text-align: center;
    }

    /* Square track boxes built from real st.container(key=...) elements so
       charts render fully nested inside the bordered tile frame */
    div[class*="st-key-trackbox"] {
        background: rgba(255,255,255,0.025) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
        padding: 0.9rem 1rem !important;
    }
    div[class*="st-key-trackbox"] [data-testid="stVerticalBlock"] {
        gap: 0.25rem;
    }

    /* Query / library panel cards built the same way for proper nesting */
    div[class*="st-key-panelcard"] {
        background: rgba(255,255,255,0.035) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 16px !important;
        padding: 1.4rem 1.5rem !important;
        margin-bottom: 1.1rem;
    }

    /* Verdict banners */
    .verdict-match {
        font-family: 'Helvetica', 'Helvetica Neue', Arial, sans-serif;
        padding: 1.05rem 1.3rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(16,185,129,0.18), rgba(16,185,129,0.05));
        border: 1px solid rgba(16,185,129,0.4);
        color: #d1fae5;
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    .verdict-match span.label {
        display: block;
        font-family: 'Courier New', Courier, monospace;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #6ee7b7;
        margin-bottom: 0.3rem;
        text-align: center;
    }
    .verdict-nomatch {
        font-family: 'Helvetica', 'Helvetica Neue', Arial, sans-serif;
        padding: 1.05rem 1.3rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(244,63,94,0.18), rgba(244,63,94,0.05));
        border: 1px solid rgba(244,63,94,0.4);
        color: #fecdd3;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    .verdict-nomatch span.label {
        display: block;
        font-family: 'Courier New', Courier, monospace;
        font-weight: 600;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #fda4af;
        margin-bottom: 0.3rem;
        text-align: center;
    }

    /* Center Aligning Nav Styling (st.segmented_control replaces st.tabs so
       switching sections is a real, lazy rerun instead of an always-rendered tab) */
    div[data-testid="stSegmentedControl"] {
        display: flex;
        justify-content: center;
        margin: 0 auto 1.5rem auto;
    }
    div[data-testid="stSegmentedControl"] > div {
        background: rgba(255,255,255,0.03);
        padding: 0.35rem;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.07);
        gap: 0.4rem;
    }
    div[data-testid="stSegmentedControl"] label {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: #b7b2cc;
        border-radius: 10px !important;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
    }
    div[data-testid="stSegmentedControl"] label[aria-checked="true"] {
        background: rgba(124,58,237,0.35) !important;
        color: #f5f3ff !important;
    }

    /* Metric tiles */
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 14px;
        padding: 0.85rem 1rem 0.7rem 1rem;
    }
    div[data-testid="stMetricLabel"] {
        color: #9893ad !important;
    }
    div[data-testid="stMetricValue"] {
        color: #f4f2ff !important;
        font-family: 'Sora', sans-serif;
    }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.03);
        border: 1.5px dashed rgba(167,139,250,0.45);
        border-radius: 14px;
    }

    /* Dataframe / table corners */
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Divider tweak */
    hr {
        border-color: rgba(255,255,255,0.08);
    }

</style>
""", unsafe_allow_html=True)

SR = 8000
DB_PATH = "song_database.npz"

@st.cache_resource(show_spinner="Memory-mapping optimized NumPy database file index scales...")
def load_numpy_database():
    if not os.path.exists(DB_PATH):
        st.error(f"Missing data registry '{DB_PATH}'! Run `python fingerprint2.py` first.")
        st.stop()
        
    data = np.load(DB_PATH, allow_pickle=True, mmap_mode='r')
    db_hashes = data["db_hashes"]
    db_songs = data["db_songs"]
    db_anchors = data["db_anchors"]
    song_names = list(data["song_names"])
    
    song_hash_counts = np.bincount(db_songs, minlength=len(song_names))
    
    song_constellations = {idx: data[f"peaks_{idx}"] if f"peaks_{idx}" in data 
                            else np.empty((0, 2), dtype=np.int32) for idx in range(len(song_names))}
    return db_hashes, db_songs, db_anchors, song_names, song_constellations, song_hash_counts

def load_query_audio(uploaded_file):
    file_bytes = io.BytesIO(uploaded_file.read())
    data, _ = librosa.load(file_bytes, sr=SR, mono=True)
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / (max_val + 1e-9)
    return data

@st.cache_data(show_spinner=False)
def render_track_thumbnail_png(track_idx, _peaks, width=380, height=130):
    """Pre-renders a single track's constellation as a flat, theme-colored PNG.

    Cached per track_idx so the pixels are computed exactly once, total -
    not once per Streamlit rerun, and not as a live JS chart engine. The
    leading underscore on `_peaks` tells Streamlit's cache to key only on
    track_idx and skip hashing the (large) array itself.
    """
    if len(_peaks) == 0:
        return None

    subsample_rate = max(1, len(_peaks) // 400)
    pts = _peaks[::subsample_rate]
    freqs, times = pts[:, 0].astype(float), pts[:, 1].astype(float)

    t_min, t_max = times.min(), times.max()
    f_min, f_max = freqs.min(), freqs.max()
    t_span = (t_max - t_min) or 1.0
    f_span = (f_max - f_min) or 1.0

    pad = 4
    xs = pad + (times - t_min) / t_span * (width - 2 * pad)
    ys = (height - pad) - (freqs - f_min) / f_span * (height - 2 * pad)

    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    r = 1.4
    fill = (167, 139, 250, 220)  # theme violet (#a78bfa) with slight transparency
    for x, y in zip(xs, ys):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# --- Static Renderers: Completely Free of Interactivity, Gridlines, and Axis Labels ---
def render_raw_static_scatter(df, x_col, y_col, color_col=None, height=180, size=15):
    """Generates an ultra-clean scatter block showing raw points only inside container contexts,
    themed to match the site's dark violet / cyan palette."""
    spec = {
        "padding": 0,
        "background": "transparent",
        "mark": {"type": "circle", "size": size, "color": "#a78bfa", "opacity": 0.85},
        "encoding": {
            "x": {
                "field": x_col, 
                "type": "quantitative", 
                "axis": {"labels": False, "grid": False, "ticks": False, "title": None, "domain": False}
            },
            "y": {
                "field": y_col, 
                "type": "quantitative", 
                "axis": {"labels": False, "grid": False, "ticks": False, "title": None, "domain": False}
            }
        },
        "config": {
            "view": {"stroke": "transparent", "fill": "transparent"},
            "background": "transparent",
            "selection": {"mesh": {"type": "disable"}}, 
        }
    }
    if color_col:
        spec["encoding"]["color"] = {
            "field": color_col, 
            "type": "nominal",
            "legend": None,
            "scale": {
                "domain": ["Library Track", "Matched Window"],
                "range": ["#4c4565", "#7dd3fc"]
            }
        }
        
    st.vega_lite_chart(df, spec, use_container_width=True, height=height)

def render_top_matches_chart(song_scores):
    if not song_scores:
        st.info("No matches registered.")
        return
        
    sorted_scores = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    df = pd.DataFrame(sorted_scores, columns=["Song Track", "Matching Votes"])
    
    st.caption("Top-5 Matching Track Distribution")
    spec = {
        "background": "transparent",
        "mark": {"type": "bar", "color": "#a78bfa", "cornerRadiusEnd": 4},
        "encoding": {
            "x": {"field": "Matching Votes", "type": "quantitative", "axis": {"grid": False}},
            "y": {"field": "Song Track", "type": "nominal", "sort": "-x"}
        },
        "config": {
            "view": {"stroke": "transparent", "fill": "transparent"},
            "background": "transparent",
            "axis": {
                "labelColor": "#b7b2cc",
                "titleColor": "#b7b2cc",
                "domainColor": "rgba(255,255,255,0.15)",
                "tickColor": "rgba(255,255,255,0.15)"
            }
        }
    }
    st.vega_lite_chart(df, spec, use_container_width=True, height=220)


# --- Main Presentation Layers ---
st.markdown("""
<div class="hero">
    <p class="hero-title">Sonic Signatures</p>
    <p class="hero-subtitle">
        A Shazam-style audio fingerprinting engine built on spectral peak constellations
        and vectorized hash matching against an indexed track library.
    </p>
    <div class="hero-pill-row">
        <span class="hero-pill">Constellation Hashing</span>
        <span class="hero-pill">Vectorized Matching</span>
        <span class="hero-pill">Offline NumPy Index</span>
    </div>
</div>
""", unsafe_allow_html=True)

db_hashes, db_songs, db_anchors, song_names, song_constellations, song_hash_counts = load_numpy_database()

NAV_OPTIONS = ["Single-Clip Identifier", "Batch Processing Mode", "Track Library Registry"]

# st.tabs() always builds every tab's content on every rerun (switching tabs is
# purely a frontend show/hide - Streamlit has no way to know which tab you're on).
# A segmented control is a real widget: picking a new section triggers a rerun
# that only executes the branch below matching that section, so the expensive
# Library grid no longer gets rebuilt in the background while you're on another tab.
active_tab = st.segmented_control(
    "Section",
    NAV_OPTIONS,
    selection_mode="single",
    default=NAV_OPTIONS[0],
    label_visibility="collapsed",
    key="active_tab",
)
if active_tab is None:
    active_tab = NAV_OPTIONS[0]

# 1. Single-Clip Section
if active_tab == "Single-Clip Identifier":
    st.markdown("""
    <p class="section-heading">Upload a Query Clip</p>
    <p class="section-caption">Drop in a short audio snippet and click start to process against the library.</p>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Clip (.mp3)", type=["mp3"], key="single_uploader", label_visibility="collapsed")

    if uploaded is not None:
        if st.button("Start Identification", type="primary", key="btn_single_start"):
            with st.spinner("Processing lightning-fast index query matching..."):
                audio = load_query_audio(uploaded)
                best_song, song_scores, q_peaks, offset, q_duration = fp.match_query(
                    audio, SR, db_hashes, db_songs, db_anchors, song_names
                )

            st.markdown("<br>", unsafe_allow_html=True)

            if best_song:
                st.markdown(f"""
                <div class="verdict-match">
                    <span class="label">Predicted Identity</span>
                    {best_song}
                </div>
                """, unsafe_allow_html=True)
                best_song_idx = song_names.index(best_song)
                target_track_peaks = song_constellations[best_song_idx]

                col1, col2 = st.columns(2, gap="medium")
                with col1:
                    with st.container(border=True, key="panelcard-query"):
                        if len(q_peaks) > 0:
                            df_q = pd.DataFrame(q_peaks, columns=["Frequency", "Time"])
                            render_raw_static_scatter(df_q, "Time", "Frequency", height=200)
                        st.divider()
                        
                        if len(target_track_peaks) > 0:
                            mask = (target_track_peaks[:, 1] >= offset) & (target_track_peaks[:, 1] <= (offset + q_duration))
                            df_lib = pd.DataFrame(target_track_peaks, columns=["Frequency", "Time"])
                            df_lib["Source"] = "Library Track"
                            df_lib.loc[mask, "Source"] = "Matched Window"
                            render_raw_static_scatter(df_lib, "Time", "Frequency", color_col="Source", height=200)
                with col2:
                    with st.container(border=True, key="panelcard-matches"):
                        render_top_matches_chart(song_scores)

# 2. Batch Processing Section
elif active_tab == "Batch Processing Mode":
    st.markdown("""
    <p class="section-heading">Batch Identification</p>
    <p class="section-caption">Upload several clips at once for sequential matching.</p>
    """, unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload Multiple Query Sequences", type=["mp3"], accept_multiple_files=True,
        key="batch_uploader", label_visibility="collapsed"
    )

    if uploaded_files:
        if st.button("Start Batch Identification", type="primary", key="btn_batch_start"):
            rows = []
            progress = st.progress(0.0)
            for idx, uf in enumerate(uploaded_files):
                try:
                    audio = load_query_audio(uf)
                    best_song, _, _, _, _ = fp.match_query(
                        audio, SR, db_hashes, db_songs, db_anchors, song_names
                    )
                    rows.append({"filename": uf.name, "prediction": best_song if best_song else "No Match"})
                except Exception as err:
                    rows.append({"filename": uf.name, "prediction": f"Error: {err}"})
                progress.progress((idx + 1) / len(uploaded_files))

            results_df = pd.DataFrame(rows, columns=["filename", "prediction"])
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(results_df, use_container_width=True)

# 3. Track Library Registry Section - only built when actually selected
elif active_tab == "Track Library Registry":
    st.markdown("""
    <p class="section-heading">Indexed Track Library</p>
    <p class="section-caption">Visual structural overview of the constellation signatures mapped inside the database index.</p>
    """, unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Indexed Tracks", len(song_names))
    with m2:
        st.metric("Total Fingerprint Hashes", f"{db_hashes.shape[0]:,}")

    st.markdown("<br><hr><br>", unsafe_allow_html=True)

    cols_per_row = 3
    for i in range(0, len(song_names), cols_per_row):
        cols = st.columns(cols_per_row, gap="medium")
        
        for next_idx in range(cols_per_row):
            track_idx = i + next_idx
            if track_idx >= len(song_names):
                break
                
            name = song_names[track_idx]
            peaks = song_constellations[track_idx]
            associated_hashes = song_hash_counts[track_idx]
            
            with cols[next_idx]:
                # Real container (not a raw markdown div) so the chart renders
                # genuinely nested inside the bordered, aspect-ratio-locked square.
                with st.container(border=True, key=f"trackbox-{track_idx}"):
                    st.markdown(
                        f'<div class="library-track-title" title="ID #{track_idx} — {name}">'
                        f'ID #{track_idx} — {name}</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f'<div class="library-track-meta">'
                        f'Peaks: <strong>{len(peaks):,}</strong> &nbsp;|&nbsp; '
                        f'Hashes: <strong>{associated_hashes:,}</strong></div>',
                        unsafe_allow_html=True
                    )

                    if len(peaks) > 0:
                        thumb_png = render_track_thumbnail_png(track_idx, peaks)
                        st.image(thumb_png, use_container_width=True)
                    else:
                        st.info("No constellation map features array found.")
