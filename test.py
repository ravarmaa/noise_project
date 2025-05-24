import os
import numpy as np
import pandas as pd
from pydub import AudioSegment

def dbfs(audio: AudioSegment):
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)
    rms = np.sqrt(np.mean(samples ** 2))
    return 20 * np.log10(rms + 1e-10)

def compute_attenuation(open_path, close_path):
    audio_open = AudioSegment.from_file(open_path)
    audio_close = AudioSegment.from_file(close_path)

    # Trim first and last 5 seconds (5000 ms)
    if len(audio_open) > 10000:
        audio_open = audio_open[5000:-5000]
    if len(audio_close) > 10000:
        audio_close = audio_close[5000:-5000]

    db_open = dbfs(audio_open)
    db_close = dbfs(audio_close)
    attenuation = db_open - db_close

    return db_open, db_close, attenuation

def process_folder(folder_path):
    results = []

    for root, dirs, files in os.walk(folder_path):
        open_file = None
        close_file = None

        for f in files:
            f_lower = f.lower()
            if "open" in f_lower and f_lower.endswith(".m4a"):
                open_file = os.path.join(root, f)
            elif "close" in f_lower and f_lower.endswith(".m4a"):
                close_file = os.path.join(root, f)

        if open_file and close_file:
            try:
                db_open, db_close, attenuation = compute_attenuation(open_file, close_file)
                results.append({
                    "Location": os.path.relpath(root, folder_path),
                    "Open File": os.path.basename(open_file),
                    "Close File": os.path.basename(close_file),
                    "dB (Open)": round(db_open, 2),
                    "dB (Close)": round(db_close, 2),
                    "Attenuation (dB)": round(attenuation, 2)
                })
            except Exception as e:
                print(f"Error processing {root}: {e}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    main_folder = "/home/stan/Documents/noise_project/curated_data"

    if not os.path.exists(main_folder):
        print(f"Folder '{main_folder}' does not exist.")
        exit()

    df = process_folder(main_folder)

    if df.empty:
        print("No valid open/close file pairs found.")
    else:
        html_path = os.path.join(main_folder, "window_attenuation_report.html")
        df.to_html(html_path, index=False)
        print(f"Report saved to {html_path}")
