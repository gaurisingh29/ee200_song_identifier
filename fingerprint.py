"""
fingerprint.py
----------------
Simple "Shazam-style" audio fingerprinting

Pipeline:
    audio -> spectrogram -> local-maxima peaks -> (peak pairs -> hashes) -> database
    query -> same pipeline -> match hashes against database -> offset histogram -> best song

"""

import os
import numpy as np
from scipy import signal as sps
from scipy.ndimage import maximum_filter
from scipy.io import wavfile


# 1. Loading audio


def load_wav(path, target_sr=11025):
    #Load a mono .wav file and resample to target_sr if needed. Returns (signal_float, sample_rate).
    sr, data = wavfile.read(path)
    if data.ndim > 1:                 # stereo -> mono
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    data = data / (np.max(np.abs(data)) + 1e-9)   # normalize to [-1, 1]

    if sr != target_sr:
        # simple resampling using scipy.signal.resample
        n_new = int(len(data) * target_sr / sr)
        data = sps.resample(data, n_new)
        sr = target_sr
    return data, sr


def list_songs(folder):
    #Return list of (name_without_ext, full_path) for .wav files in folder.
    if not os.path.isdir(folder):
        return []
    out = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(".wav"):
            out.append((os.path.splitext(f)[0], os.path.join(folder, f)))
    return out


# 2. Spectrogram

def spectrogram(audio, sr, nperseg=1024, noverlap=None):
    """Thin wrapper around scipy.signal.spectrogram.
    Returns f (Hz), t (s), Sxx (magnitude, NOT dB)."""
    if noverlap is None:
        noverlap = nperseg // 2
    f, t, Sxx = sps.spectrogram(audio, fs=sr, window='hann',
                                 nperseg=nperseg, noverlap=noverlap,
                                 mode='magnitude')
    return f, t, Sxx


# 3. Peak picking ("constellation map")

def find_peaks_2d(Sxx, amp_min_db=-25, neighborhood=(20, 5)):
    """Find local maxima in a spectrogram.

    Sxx           : magnitude spectrogram (freq_bins x time_bins)
    amp_min_db    : ignore peaks weaker than this, relative to the loudest point (dB)
    neighborhood  : (freq_window, time_window) size of the local max filter

    Returns list of (freq_bin_index, time_bin_index).
    """
    Sxx_db = 20 * np.log10(Sxx + 1e-9)
    Sxx_db -= Sxx_db.max()              # 0 dB = loudest point in this clip

    local_max = maximum_filter(Sxx_db, size=neighborhood) == Sxx_db
    above_thresh = Sxx_db > amp_min_db
    peak_mask = local_max & above_thresh

    freq_idx, time_idx = np.where(peak_mask)
    peaks = list(zip(freq_idx, time_idx))
    return peaks


# ----------------------------------------------------------------------
# 4. Hashing: pair nearby peaks
# ----------------------------------------------------------------------

def generate_hashes(peaks, fan_out=5, min_dt=1, max_dt=50):
    """Turn a list of (freq_idx, time_idx) peaks into hashes.

    Each peak is paired with up to `fan_out` other peaks that come shortly
    after it in time (a "target zone"), forming a hash:
        hash_key   = (f1, f2, dt)        -- two frequencies + time gap
        anchor_time= t1                  -- time of the FIRST peak in the pair

    This is the same idea Shazam uses: a single peak is easily confused with
    noise, but a *pair* of peaks with a specific frequency/time relationship
    is a much more unique "fingerprint".
    """
    peaks = sorted(peaks, key=lambda p: p[1])   # sort by time
    hashes = []   # list of (hash_key, anchor_time)

    for i, (f1, t1) in enumerate(peaks):
        count = 0
        for f2, t2 in peaks[i + 1:]:
            dt = t2 - t1
            if dt < min_dt:
                continue
            if dt > max_dt:
                break                  # peaks are time-sorted, no point going further
            hashes.append(((f1, f2, dt), t1))
            count += 1
            if count >= fan_out:
                break
    return hashes


def generate_single_peak_hashes(peaks):
    #Alternative (weaker) fingerprint: use each peak's frequency alone as the hash key without pairing.
    return [((f1,), t1) for (f1, t1) in peaks]


# 5. Database build + matching

def build_database(songs, nperseg=1024, fan_out=5, use_pairs=True):
    """songs: list of (name, audio, sr). Returns dict: hash_key -> list of (song_name, anchor_time)."""
    db = {}
    for name, audio, sr in songs:
        f, t, Sxx = spectrogram(audio, sr, nperseg=nperseg)
        peaks = find_peaks_2d(Sxx)
        hashes = generate_hashes(peaks, fan_out=fan_out) if use_pairs \
            else generate_single_peak_hashes(peaks)
        for h, anchor_t in hashes:
            db.setdefault(h, []).append((name, anchor_t))
    return db


def match_query(audio, sr, db, nperseg=1024, fan_out=5, use_pairs=True):
    """Identify `audio` against database `db`.

    Returns (best_song_name_or_None, offset_histograms_dict)
    offset_histograms_dict[song_name] = {offset: count}, useful for plotting.
    """
    f, t, Sxx = spectrogram(audio, sr, nperseg=nperseg)
    peaks = find_peaks_2d(Sxx)
    q_hashes = generate_hashes(peaks, fan_out=fan_out) if use_pairs \
        else generate_single_peak_hashes(peaks)

    offset_counts = {}   # song_name -> {offset: count}
    for h, q_time in q_hashes:
        if h in db:
            for song_name, db_time in db[h]:
                offset = db_time - q_time     # how much the query is shifted vs. the DB track
                offset_counts.setdefault(song_name, {})
                offset_counts[song_name][offset] = offset_counts[song_name].get(offset, 0) + 1

    best_song, best_score = None, 0
    for song_name, hist in offset_counts.items():
        peak_score = max(hist.values())
        if peak_score > best_score:
            best_score = peak_score
            best_song = song_name

    return best_song, offset_counts, peaks


# ----------------------------------------------------------------------
# 6. Synthetic song generator (ONLY used if you have not yet supplied the
#    real song library handed out in the course). Lets the whole pipeline
#    be tested end-to-end. Replace with real .wav files for submission.
# ----------------------------------------------------------------------

def make_synthetic_song(name, sr=11025, duration=6.0, seed=0):
    """Create a simple synthetic 'melody': a sequence of tones (sine waves)
    at different frequencies, so different synthetic songs are distinguishable,
    just like real songs have distinguishable spectral content over time."""
    rng = np.random.RandomState(seed)
    n = int(sr * duration)
    t = np.arange(n) / sr
    audio = np.zeros(n)

    note_len = sr // 2  # half-second notes
    n_notes = n // note_len
    base_freqs = rng.choice([220, 261, 294, 330, 392, 440, 523], size=n_notes)

    for i, f0 in enumerate(base_freqs):
        seg = slice(i * note_len, (i + 1) * note_len)
        tt = t[seg] - t[seg][0]
        # fundamental + a couple of harmonics, like a real instrument
        tone = (np.sin(2 * np.pi * f0 * tt)
                + 0.5 * np.sin(2 * np.pi * 2 * f0 * tt)
                + 0.25 * np.sin(2 * np.pi * 3 * f0 * tt))
        envelope = np.hanning(len(tt))     # smooth note on/off (avoids clicks)
        audio[seg] += tone * envelope

    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    return audio


def add_noise(audio, snr_db):
    """Add white Gaussian noise at a given SNR (dB)."""
    sig_power = np.mean(audio ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), size=audio.shape)
    return audio + noise


def pitch_shift_resample(audio, sr, semitones):
    """Crude pitch shift: resample then either pad/trim back to original length.
    (Changes both pitch AND tempo, like the classic 'speed up a tape' trick -
    simple, but enough to demonstrate the fingerprinting failure mode.)"""
    factor = 2 ** (semitones / 12.0)
    n_new = int(len(audio) / factor)
    shifted = sps.resample(audio, n_new)
    return shifted
