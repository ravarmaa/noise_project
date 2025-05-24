import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

from scipy.signal import sosfilt, bilinear_zpk, zpk2sos

def a_weighting_filter(fs):
    if fs != 44100:
        raise ValueError("Only 44100 Hz sample rate is currently supported for A-weighting.")
    # Pre-computed A-weighting filter coefficients for 44100 Hz
    # Source: Librosa, standard A-weighting implementation
    sos = np.array([
        [0.16999495, -0.3399899, 0.16999495, 1.0, -1.7600419, 0.80258177],
        [1.0, -2.0, 1.0, 1.0, -1.9501776, 0.95124613],
        [1.0, -2.0, 1.0, 1.0, -1.9660543, 0.96729257]
    ])
    return sos

def apply_a_weighting(samples, fs):
    sos = a_weighting_filter(fs)
    return sosfilt(sos, samples)

def dbfs(samples):
    rms = np.sqrt(np.mean(samples ** 2))
    return 20 * np.log10(rms + 1e-12)

# === Load your phone recording ===
path = "curated_data/max_volume.m4a"
audio = AudioSegment.from_file(path)
samples = np.array(audio.get_array_of_samples()).astype(np.float32)
samples /= float(2 ** (audio.sample_width * 8 - 1))
if audio.channels > 1:
    samples = samples.reshape((-1, audio.channels)).mean(axis=1)
fs = audio.frame_rate

# === Compute average dBFS over the full clip ===
print(f"Sample rate: {fs}")
print("Raw RMS:", np.sqrt(np.mean(samples**2)))

weighted = apply_a_weighting(samples, fs)
print("A-weighted RMS:", np.sqrt(np.mean(weighted**2)))
print("A-weighted dBFS:", dbfs(weighted))

measured_avg_dBFS = dbfs(weighted)
offset = 73.5 - measured_avg_dBFS

print(f"Average dBFS: {measured_avg_dBFS:.2f}, Offset: {offset:.2f} dB")


# === Per-second dB(A) using the offset ===
interval = fs
dba_values = []
timestamps = []

for i in range(0, len(samples), interval):
    chunk = apply_a_weighting(samples[i:i + interval], fs)
    if len(chunk) < interval:
        break
    dba_values.append(dbfs(chunk) + offset)

    timestamps.append(i / fs)

# === Plot ===
plt.figure(figsize=(10, 5))
plt.plot(timestamps, dba_values, marker="o")
plt.xlabel("Time (s)")
plt.ylabel("Estimated dB(A)")
plt.title("dB(A) Over Time (Calibrated from Full Clip Average)")
plt.grid(True)
plt.tight_layout()
plt.show()
