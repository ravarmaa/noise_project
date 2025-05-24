import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.signal import sosfilt, bilinear_zpk, zpk2sos
import csv

# === A-weighting filter ===
def a_weighting_filter(fs):
    if fs == 44100:
        return np.array([
            [0.16999495, -0.3399899, 0.16999495, 1.0, -1.7600419, 0.80258177],
            [1.0, -2.0, 1.0, 1.0, -1.9501776, 0.95124613],
            [1.0, -2.0, 1.0, 1.0, -1.9660543, 0.96729257]
        ])
    elif fs == 48000:
        return np.array([
            [0.1752076, -0.3504152, 0.1752076, 1.0, -1.7715237, 0.8044141],
            [1.0, -2.0, 1.0, 1.0, -1.9583534, 0.9600623],
            [1.0, -2.0, 1.0, 1.0, -1.9708162, 0.9725126]
        ])
    else:
        raise ValueError(f"A-weighting not supported for fs={fs}")

def apply_a_weighting(samples, fs):
    sos = a_weighting_filter(fs)
    return sosfilt(sos, samples)

def dbfs(samples):
    rms = np.sqrt(np.mean(samples ** 2))
    return 20 * np.log10(rms + 1e-12)

# === Load .m4a audio ===
audio_path = "curated_data/50.m4a"
audio = AudioSegment.from_file(audio_path)
samples = np.array(audio.get_array_of_samples()).astype(np.float32)
samples /= float(2 ** (audio.sample_width * 8 - 1))
if audio.channels > 1:
    samples = samples.reshape((-1, audio.channels)).mean(axis=1)
fs = audio.frame_rate

# === Compute A-weighted dBFS per second ===
interval = fs  # 1-second chunks
timestamps = []
weighted_dbfs = []

for i in range(0, len(samples) - interval + 1, interval):
    chunk = samples[i:i + interval]
    weighted_chunk = apply_a_weighting(chunk, fs)
    weighted_dbfs.append(dbfs(weighted_chunk))
    timestamps.append(i / fs)

# === Load Decibel X data ===
decibelx_path = "curated_data/decibelx.txt"
decibelx_dba = []
decibelx_time = []

with open(decibelx_path, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) != 2:
            continue
        try:
            val = float(row[0])
            ts = int(row[1])
            decibelx_dba.append(val)
            decibelx_time.append(ts)
        except:
            continue

# Normalize Decibel X timestamps
if decibelx_time:
    t0 = decibelx_time[0]
    decibelx_time = [(t - t0) / 1000 for t in decibelx_time]

# === Compute average offset ===
if weighted_dbfs and decibelx_dba:
    avg_weighted = np.mean(weighted_dbfs)
    avg_decibelx = np.mean(decibelx_dba)
    offset = avg_decibelx - avg_weighted
    print(f"Average A-weighted dBFS: {avg_weighted:.2f}")
    print(f"Average Decibel X dBA: {avg_decibelx:.2f}")
    print(f"Calculated offset: {offset:.2f} dB")
else:
    print("Error: missing data to compute offset.")
    exit()

# === Apply offset to weighted dBFS
calibrated_dba = [x + offset for x in weighted_dbfs]

# === Plot ===
plt.figure(figsize=(12, 6))
plt.plot(timestamps, calibrated_dba, label="Estimated dB(A) from 50.m4a", marker="o")
plt.plot(decibelx_time, decibelx_dba, label="Decibel X dB(A)", marker="x")
plt.xlabel("Time (s)")
plt.ylabel("dB(A)")
plt.title("Calibrated A-weighted dBFS vs. Decibel X")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
