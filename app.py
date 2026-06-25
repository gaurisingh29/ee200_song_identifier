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
    page_title= "Melody Matcher",
    page_icon=":material/graphic_eq:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# long long html css design 
st.markdown("""
<style>

    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

    /* ── CSS custom properties (light mode defaults) ── */
    :root {
        --bg-base:        #fdf4f8;
        --bg-surface:     #ffffff;
        --bg-surface2:    #fef6fb;
        --border:         #f0d6e8;
        --border-strong:  #e0b8d4;
        --text-primary:   #1e0a2e;
        --text-secondary: #7a4d6a;
        --text-muted:     #b07a9a;
        --accent:         #6d28d9;
        --accent-light:   #ede9fe;
        --accent-glow:    rgba(109,40,217,0.18);
        --tab-bg:         #f3e8f9;
        --tab-border:     #d8b4e8;
        --shadow-sm:      0 2px 8px rgba(109,40,217,0.07);
        --shadow-md:      0 6px 24px rgba(109,40,217,0.10);
    }

    /* Dark mode overrides */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-base:        #110820;
            --bg-surface:     #1c0f2e;
            --bg-surface2:    #231240;
            --border:         #3a1f5c;
            --border-strong:  #5a3080;
            --text-primary:   #f0e6ff;
            --text-secondary: #c4a8e0;
            --text-muted:     #8a6aaa;
            --accent:         #a78bfa;
            --accent-light:   #2d1b5e;
            --accent-glow:    rgba(167,139,250,0.22);
            --tab-bg:         #2a1650;
            --tab-border:     #5a3090;
            --shadow-sm:      0 2px 8px rgba(0,0,0,0.4);
            --shadow-md:      0 6px 24px rgba(0,0,0,0.5);
        }
    }

    /* Streamlit dark-mode class override (for in-app toggle) */
    [data-theme="dark"] {
        --bg-base:        #110820 !important;
        --bg-surface:     #1c0f2e !important;
        --bg-surface2:    #231240 !important;
        --border:         #3a1f5c !important;
        --border-strong:  #5a3080 !important;
        --text-primary:   #f0e6ff !important;
        --text-secondary: #c4a8e0 !important;
        --text-muted:     #8a6aaa !important;
        --accent:         #a78bfa !important;
        --accent-light:   #2d1b5e !important;
        --accent-glow:    rgba(167,139,250,0.22) !important;
        --tab-bg:         #2a1650 !important;
        --tab-border:     #5a3090 !important;
        --shadow-sm:      0 2px 8px rgba(0,0,0,0.4) !important;
        --shadow-md:      0 6px 24px rgba(0,0,0,0.5) !important;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: var(--text-primary);
    }

    .stApp {
        background: var(--bg-base) !important;
    }

    header[data-testid="stHeader"] { background: transparent; }

    /* ── Hero banner ── */
    .hero {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1rem;
        border-radius: 20px;
        background: var(--bg-surface);
        border: 1px solid var(--border);
        box-shadow: var(--shadow-md), 0 0 0 4px var(--accent-glow);
        margin-bottom: 2rem;
        width: fit-content;
        margin-left: auto;
        margin-right: auto;
    }
    .hero-banner-img {
        width: 100%;
        max-width: 650px;
        border-radius: 12px;
        display: block;
    }

    /* ── Typography ── */
    .library-track-title {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 1.0rem;
        color: var(--text-primary);
        margin-bottom: 0.2rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        text-align: center;
    }
    .library-track-meta {
        font-family: 'Courier New', Courier, monospace;
        font-size: 0.82rem;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .library-track-meta strong { color: var(--accent); }

    .section-heading {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
        font-size: 1.18rem;
        color: var(--text-primary);
        margin-bottom: 0.15rem;
        text-align: center;
    }
    .section-caption {
        font-family: 'Courier New', Courier, monospace;
        color: var(--text-secondary);
        font-size: 0.9rem;
        margin-bottom: 1.05rem;
        text-align: center;
    }

    /* ── Track boxes ── */
    div[class*="st-key-trackbox"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
        padding: 0.9rem 1rem !important;
        box-shadow: var(--shadow-sm);
    }
    div[class*="st-key-trackbox"] [data-testid="stVerticalBlock"] { gap: 0.25rem; }

    /* ── Panel cards ── */
    div[class*="st-key-panelcard"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        padding: 1.4rem 1.5rem !important;
        margin-bottom: 1.1rem;
        box-shadow: var(--shadow-sm);
    }

    /* ── Verdict banners ── */
    .verdict-match {
        font-family: 'Sora', sans-serif;
        padding: 1.05rem 1.3rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #d1fae5, #f0fdf4);
        border: 1px solid #10b981;
        color: #065f46;
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
        color: #047857;
        margin-bottom: 0.3rem;
    }
    .verdict-nomatch {
        font-family: 'Sora', sans-serif;
        padding: 1.05rem 1.3rem;
        border-radius: 14px;
        background: linear-gradient(135deg, #fee2e2, #fef2f2);
        border: 1px solid #ef4444;
        color: #991b1b;
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
        color: #b91c1c;
        margin-bottom: 0.3rem;
    }

    /* ── Segmented control (tabs) — CENTERED ── */
    div[data-testid="stSegmentedControl"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        margin: 0 auto 1.5rem auto !important;
    }
    div[data-testid="stSegmentedControl"] > div[role="radiogroup"] {
        display: inline-flex !important;
        justify-content: center !important;
        align-items: center !important;
        background: var(--tab-bg) !important;
        padding: 0.35rem !important;
        border-radius: 16px !important;
        border: 1px solid var(--tab-border) !important;
        gap: 0.35rem !important;
        margin: 0 auto !important;
        box-shadow: var(--shadow-sm);
    }
    div[data-testid="stSegmentedControl"] label {
        font-family: 'Sora', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        color: var(--text-secondary) !important;
        border-radius: 11px !important;
        padding: 0.4rem 1.25rem !important;
        transition: all 0.18s ease !important;
        letter-spacing: 0.01em !important;
    }
    div[data-testid="stSegmentedControl"] label[aria-checked="true"] {
        background: var(--accent) !important;
        color: #ffffff !important;
        box-shadow: 0 3px 14px var(--accent-glow) !important;
    }

    /* ── Metric tiles ── */
    div[data-testid="stMetric"] {
        background: var(--bg-surface);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 0.85rem 1rem 0.7rem 1rem;
        box-shadow: var(--shadow-sm);
    }
    div[data-testid="stMetricLabel"] { color: var(--text-secondary) !important; }
    div[data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-family: 'Sora', sans-serif;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--bg-surface2);
        border: 1.5px dashed var(--border-strong);
        border-radius: 14px;
    }

    /* ── Misc ── */
    div[data-testid="stDataFrame"], div[data-testid="stTable"] {
        border-radius: 12px;
        overflow: hidden;
    }
    hr { border-color: var(--border); }

</style>
""", unsafe_allow_html=True)

# actual application structure

SR = 8000 #lower frequencies had more peaks, so trimmed out the sampling to exclude higher frequencies (to reduce size of the database)
DB_PATH = "song_database.npz"

@st.cache_resource(show_spinner="Memory-mapping optimized NumPy database file index scales...") #take the entire db into memory 
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


#calculating it from the db was a better idea than directly computing the spectrogram again 
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
    fill = (124, 58, 237, 220)  # theme violet (#a78bfa) with slight transparency
    for x, y in zip(xs, ys):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

#rendered static images rather than actual plots, so that there is no additional overhead and memory utilization
#used vega-lite for plotting, as it was more lightweight compared to plotly and matplotlib and native streamlit plot
def render_raw_static_scatter(df, x_col, y_col, color_col=None, height=180, size=15):
    """Generates an ultra-clean scatter block showing raw points only inside container contexts,
    themed to match the site's dark violet / cyan palette."""
    spec = {
        "padding": 0,
        "background": "transparent",
        "mark": {"type": "circle", "size": size, "color": "#7c3aed", "opacity": 0.85},
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
                "range": ["#cbd5e1", "#0284c7"]
            }
        }
        
    st.vega_lite_chart(df, spec, use_container_width=True, height=height)



#charts for the query matching display
def render_top_matches_chart(song_scores):
    if not song_scores:
        st.info("No matches registered.")
        return
        
    sorted_scores = sorted(song_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    df = pd.DataFrame(sorted_scores, columns=["Song Track", "Matching Votes"])
    
    st.caption("Top-5 Matching Track Distribution")
    spec = {
        "background": "transparent",
        "mark": {"type": "bar", "color": "#7c3aed", "cornerRadiusEnd": 4},
        "encoding": {
            "x": {"field": "Matching Votes", "type": "quantitative", "axis": {"grid": False}},
            "y": {"field": "Song Track", "type": "nominal", "sort": "-x"}
        },
        "config": {
            "view": {"stroke": "transparent", "fill": "transparent"},
            "background": "transparent",
            "axis": {
                "labelColor": "#475569",
                "titleColor": "#475569",
                "domainColor": "rgba(0,0,0,0.15)",
                "tickColor": "rgba(0,0,0,0.15)"
            }
        }
    }
    st.vega_lite_chart(df, spec, use_container_width=True, height=220)



import base64

# function to safely load a local image and convert it to base64
def get_b64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

# Encode your local header image
img_b64 = get_b64_image("./header_image.jpg")

#actual applicaton part, other than the defined function above
st.markdown(f"""
<div class="hero">
    <img src="data:image/jpeg;base64,{img_b64}" class="hero-banner-img" alt="Melody Matcher Banner">
</div>
""", unsafe_allow_html=True)

db_hashes, db_songs, db_anchors, song_names, song_constellations, song_hash_counts = load_numpy_database()

NAV_OPTIONS = ["Single-Clip Identifier", "Batch Processing Mode", "Track Library Registry"]

# st.tabs() builds every tab's content on every tabswitch/rerun
# segmented control is a real widget: picking a new section triggers a rerun
# did this switch from traditional st.tabs() because, the regeneration library was slow
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

# single clip section
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

# batch process section
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
            
            st.session_state["batch_results"] = results_df
    if "batch_results" in st.session_state:
        df_to_show = st.session_state["batch_results"]
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_to_show, use_container_width=True)
        
        csv_buffer = df_to_show.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="Download Results as CSV",
            data=csv_buffer,
            file_name="results.csv",
            mime="text/csv",
            use_container_width=True
        )

# track library section (took a lot of effort to make it pretty and fast)
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

## Current modules app and fingerprint are overoptimized and overengineered
## Tried implementation using much less UI/UX, with only query processing using sqlite
## time taken for that particular implementation was ~150-200 ms on first load
## while this implementation, loads ~5-10 ms, though not significant for this database, when extended
## to a larger database would perform gay would be clearly visible. 

## the indexing of the current database is limited to 256 at max, as uint_8 is used for array indexing
