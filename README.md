# Melody Matcher – Audio Fingerprinting System

A lightweight, high-performance audio fingerprinting system for identifying songs from short audio clips. The project combines a NumPy-optimized fingerprint database with a Streamlit interface to provide fast single-query and batch audio identification. The implementation focuses on minimizing CPU overhead, memory usage, and repeated computations while maintaining scalability for larger music libraries.  


## Usage
1. First install the required packages
`pip install -r requirements.txt `
2. Then, during the first run, to generate the database, after placing the songs to be databased at `./songs/`
`python fingerprint.py `
3. To get the interface, run 
`streamlit run app.py` 

## Program Flow

### 1. Database Construction (Offline)

The fingerprint module scans every `.mp3` file inside the `songs/` directory and processes each song only once.

Pipeline:

1. Load and normalize audio.
2. Generate an STFT spectrogram.
3. Detect local spectral peaks.
4. Form constellation maps from the detected peaks.
5. Generate fingerprint hashes using pairs of nearby peaks.
6. Store all hashes, anchor timestamps, song indices and constellation maps inside a compressed NumPy database (`song_database.npz`).

The database is sorted by fingerprint hash before saving, allowing efficient binary-search based retrieval during querying instead of scanning every fingerprint. 

## Query Processing

When a user uploads an audio clip:

1. Audio is loaded and normalized.
2. The same fingerprint generation pipeline is executed.
3. Query hashes are matched against the pre-built database.
4. Matching fingerprint offsets are accumulated as votes.
5. The song receiving the maximum consistent votes is returned as the predicted match.
6. The UI visualizes the query constellation, matched region, and top candidate scores.  

## Optimizations

### Reduced Sampling Rate

The system uses an **8 kHz sampling rate** instead of full-quality audio.

**Reason**

* Higher frequencies contribute little to fingerprint uniqueness.
* Reduces spectrogram size.
* Produces fewer peaks and hashes.
* Lowers memory and computation cost without significantly affecting identification accuracy. 

### NumPy Database

Instead of SQLite or Redis, fingerprints are stored inside a compressed NumPy archive.

**Reason**

* Data is already required as arrays.
* Enables vectorized operations.
* Eliminates SQL parsing and indexing overhead.
* Supports direct memory mapping. 

### Memory-Mapped Loading

The fingerprint database is loaded using `np.load(..., mmap_mode='r')`.

**Reason**

* Prevents unnecessary copying of large arrays.
* Allows fast startup.
* Scales better as the database grows. 

### Compact Integer Storage

Different integer types are selected according to data range:

* `uint64` for packed hashes
* `uint8` for song IDs (≤256 songs)
* `int32` for timestamps

**Reason**

* Smaller memory footprint.
* Better cache locality.
* Faster array operations.  

### Packed Fingerprint Hashes

Each fingerprint is packed into a single 64-bit integer containing two frequency indices and the time difference.

**Reason**

* Compact storage.
* Fast integer comparisons.
* Efficient sorting and searching. 

### Binary Search Matching

Query fingerprints are matched using `numpy.searchsorted()` on the sorted database.

**Reason**

* Replaces linear scanning.
* Performs logarithmic-time lookup.
* Greatly reduces query latency on large databases. 

### Vectorized Vote Aggregation

Matching offsets and song IDs are processed using NumPy vectorized operations instead of Python loops.

**Reason**

* Minimizes interpreter overhead.
* Improves CPU utilization.
* Enables efficient processing of thousands of matches simultaneously. 

### Streamlit Caching

Frequently reused resources such as the fingerprint database and track thumbnails are cached.

**Reason**

* Avoids repeated computation on every UI refresh.
* Improves responsiveness during repeated queries. 

### Lightweight Visualization

Instead of interactive plotting libraries for every track, the application pre-renders constellation thumbnails and uses Vega-Lite for lightweight scatter plots.

**Reason**

* Lower rendering overhead.
* Reduced memory consumption.
* Faster dashboard loading while preserving visual information. 

## Features

* Single audio clip identification
* Batch processing of multiple clips
* Precomputed fingerprint database
* Interactive visualization of constellation maps
* CSV export for batch predictions
* Optimized fingerprint indexing and matching pipeline designed for efficient scaling to larger music libraries. 

