import os
import numpy as np
import pandas as pd
from pydub import AudioSegment
import matplotlib.pyplot as plt
from mutagen.mp4 import MP4
from jinja2 import Template
import json
from scipy.signal import bilinear_zpk, zpk2sos, sosfilt

def apply_a_weighting(samples, fs):
    sos = a_weighting_filter(fs)
    return sosfilt(sos, samples)

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
        raise ValueError(f"A-weighting filter not supported for fs = {fs}. Supported: 44100, 48000 Hz.")


def dbfs(samples):
    rms = np.sqrt(np.mean(samples ** 2))
    return 20 * np.log10(rms + 1e-12)

def extract_metadata(file_path):
    """
    Extracts metadata from the .m4a file, including creation time and other tags.
    """
    try:
        audio = MP4(file_path)
        creation_time = audio.tags.get("\xa9day", ["Unknown"])[0]
        name = audio.tags.get("©nam", ["Unknown"])[0]
        device_info = audio.tags.get("©too", ["Unknown"])[0]
        return creation_time, name, device_info
    except Exception as e:
        print(f"Error extracting metadata from {file_path}: {e}")
        return "Unknown", "Unknown", "Unknown"


def process_calibration_files(folder_path, output_folder, reference_dba=73.5):
    """
    Process calibration files: compute dBFS, estimate offsets from peak,
    and plot both raw dBFS and calibrated dB(A) signals.
    """
    calibration_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower() in {"calibration.m4a", "calibration.mp4"}:
                calibration_files.append(os.path.join(root, f))

    if not calibration_files:
        print("No calibration files found.")
        return {}, None, None

    signals = []  # (file_path, dbfs_levels, timestamps)
    offsets = {}

    for file_path in calibration_files:
        audio = AudioSegment.from_file(file_path)
        fs = audio.frame_rate
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        bit_depth = audio.sample_width * 8
        samples /= float(2 ** (bit_depth - 1))
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels)).mean(axis=1)

        # Divide into 1s segments and compute dBFS
        interval = fs
        dbfs_levels = []
        timestamps = []
        for i in range(0, len(samples) - interval + 1, interval):
            chunk = samples[i:i + interval]
            dbfs_levels.append(dbfs(chunk))
            timestamps.append(i / fs)

        # Compute calibration offset
        top5 = sorted(dbfs_levels, reverse=True)[:5]
        peak_dbfs = np.mean(top5)
        offset = reference_dba - peak_dbfs
        offsets[os.path.basename(os.path.dirname(file_path))] = offset
        signals.append((file_path, dbfs_levels, timestamps, offset))


    offset_avg = np.mean(list(offsets.values()))
    offsets["average"] = offset_avg

    # Plot raw dBFS
    before_plot_path = os.path.join(output_folder, "calibration_dbfs_plot.png")
    plt.figure(figsize=(12, 6))
    for file_path, dbfs_levels, timestamps, _ in signals:
        plt.plot(timestamps, dbfs_levels, label=os.path.basename(os.path.dirname(file_path)))
    plt.legend()
    plt.title("Calibration dBFS Signals")
    plt.xlabel("Time (s)")
    plt.ylabel("dBFS (Unweighted)")
    plt.grid()
    plt.savefig(before_plot_path)
    plt.close()

    # Plot calibrated dB(A)
    after_plot_path = os.path.join(output_folder, "calibration_dba_plot.png")
    plt.figure(figsize=(12, 6))
    for file_path, dbfs_levels, timestamps, offset in signals:
        calibrated_dba = [v + offset for v in dbfs_levels]
        plt.plot(timestamps, calibrated_dba, label=os.path.basename(os.path.dirname(file_path)))
    plt.legend()
    plt.title("Calibration Estimated dB(A) Signals")
    plt.xlabel("Time (s)")
    plt.ylabel("Estimated dB(A)")
    plt.grid()
    plt.savefig(after_plot_path)
    plt.close()

    print(f"Calibration plots saved to {before_plot_path} and {after_plot_path}")
    return offsets, before_plot_path, after_plot_path

def process_file(file_path, calibration_offset=0.0, output_folder=None):
    """
    Processes a single audio file and plots raw dBFS, A-weighted dBFS, and calibrated dB(A).
    """
    audio = AudioSegment.from_file(file_path)
    fs = audio.frame_rate
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    bit_depth = audio.sample_width * 8
    samples /= float(2 ** (bit_depth - 1))
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)

    raw_dbfs_levels = []
    weighted_dbfs_levels = []
    calibrated_dba_levels = []
    timestamps = []

    step = fs  # 1-second intervals


    from scipy.fft import rfft, rfftfreq
    chunk = samples[:fs*60]
    spectrum = 20 * np.log10(np.abs(rfft(chunk)) + 1e-10)
    freqs = rfftfreq(len(chunk), d=1/fs)

    plt.plot(freqs, spectrum)
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.title("Spectrum of Raw Signal")
    plt.grid()
    plt.show()

    for t in range(0, len(samples) - step + 1, step):
        chunk = samples[t:t + step]
        weighted_chunk = apply_a_weighting(chunk, fs)

        


        raw = dbfs(chunk)
        weighted = dbfs(weighted_chunk)
        calibrated = weighted + calibration_offset

        raw_dbfs_levels.append(raw)
        weighted_dbfs_levels.append(weighted)
        calibrated_dba_levels.append(calibrated)
        timestamps.append(t / fs)

    # --- Plot
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, raw_dbfs_levels, label="Raw dBFS", linestyle="--")
    plt.plot(timestamps, weighted_dbfs_levels, label="A-weighted dBFS", linestyle=":")
    plt.plot(timestamps, calibrated_dba_levels, label="Calibrated dB(A)", linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Level (dB)")
    plt.title(f"Signal Levels – {os.path.basename(file_path)}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_folder:
        base = os.path.splitext(os.path.basename(file_path))[0]
        plot_path = os.path.join(output_folder, f"{base}_dba_debug_plot.png")
        plt.savefig(plot_path)
        print(f"Saved plot: {plot_path}")
    else:
        plt.show()

    plt.close()

    return timestamps, calibrated_dba_levels

def process_file2(file_path, calibration_offset=0.0):
    audio = AudioSegment.from_file(file_path)
    duration = len(audio)  # in milliseconds
    step = 1000
    timestamps = []
    dba_levels = []

    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    bit_depth = audio.sample_width * 8
    samples /= float(2 ** (bit_depth - 1))
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)
    fs = audio.frame_rate
    weighted = apply_a_weighting(samples, fs)

    for t in range(0, duration, step):
        start = int(t * fs / 1000)
        end = int((t + step) * fs / 1000)
        dba_levels.append(dbfs(weighted[start:end]) + calibration_offset)
        timestamps.append(t / 1000)

    discard = 15 * 1000
    valid_start = discard
    valid_end = duration - discard

    if "Margot" in file_path:
        valid_start = 3000 * 1000

    valid_timestamps = []
    valid_dba_levels = []

    for t in range(valid_start, valid_end, step):
        start = int(t * fs / 1000)
        end = int((t + step) * fs / 1000)
        valid_dba_levels.append(dbfs(weighted[start:end]) + calibration_offset)
        valid_timestamps.append(t / 1000)

    discarded_percentage = ((valid_start + (duration - valid_end)) / duration) * 100

    return timestamps, dba_levels, valid_timestamps, valid_dba_levels, discarded_percentage


def process_folder(data_folder, output_folder):
    results = []
    plots = []
    valid_ranges = {}
    total_raw_minutes = 0
    total_good_minutes = 0

    # Process calibration files
    calibration_offsets, calibration_before_plot, calibration_after_plot = process_calibration_files(data_folder, output_folder)

    for root, dirs, files in os.walk(data_folder):
        # Determine the relative path of the current folder
        relative_path = os.path.relpath(root, data_folder)
        # Create the corresponding subfolder in the output folder
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        # Check for device.txt in the current folder
        device_info = "Unknown"
        device_file_path = os.path.join(root, "device.txt")
        if os.path.exists(device_file_path):
            try:
                with open(device_file_path, "r") as device_file:
                    device_info = device_file.read().strip()
            except Exception as e:
                print(f"Error reading {device_file_path}: {e}")

        for f in files:
            f_lower = f.lower()
            if (f_lower.endswith(".m4a") or f_lower.endswith(".mp4")) and \
               "open" not in f_lower and "close" not in f_lower and "calibration" not in f_lower:
                file_path = os.path.join(root, f)
                try:
                    # Process the file
                    # Get calibration offset for this file
                    
                    author = os.path.basename(os.path.dirname(file_path))
                    offset = calibration_offsets.get(author, calibration_offsets.get("average", 0.0))
            
                    if "Rando" not in author:
                        print(f"Skipping {file_path} as it is not a Rando recording.")
                        continue
                    # Process the file using the offset
                    print(f"Processing {author} with offset {offset} dB")
                    timestamps, dba_levels, valid_timestamps, valid_dba_levels, discarded_percentage = process_file(file_path, calibration_offset=offset)

                    avg_dba = np.mean(dba_levels)
                    avg_valid_dba = np.mean(valid_dba_levels)

                    # Calculate raw and good data in minutes
                    raw_minutes = len(timestamps) / 60
                    good_minutes = len(valid_timestamps) / 60
                    total_raw_minutes += raw_minutes
                    total_good_minutes += good_minutes

                    # Save valid range
                    valid_ranges[os.path.relpath(file_path, data_folder)] = {
                        "Valid Start (s)": valid_timestamps[0],
                        "Valid End (s)": valid_timestamps[-1],
                        "Discarded (%)": round(discarded_percentage, 2)
                    }

                    results.append({
                        "File": os.path.relpath(file_path, data_folder),
                        "Device Info": device_info,
                        "Average dB(A)": round(avg_dba, 2),
                        "Average Valid dB(A)": round(avg_valid_dba, 2),
                        "Discarded (%)": round(discarded_percentage, 2)
                    })

                    # Plot dB(A) levels
                    plt.figure(figsize=(10, 6))
                    plt.plot(timestamps, dba_levels, label="dB(A) Levels")
                    plt.xlabel("Time (s)")
                    plt.ylabel("dB(A)")
                    plt.title(f"dB(A) Levels for {os.path.basename(file_path)}")
                    plt.legend()
                    plt.grid()
                    dba_plot_path = os.path.join(output_subfolder, f"{os.path.splitext(f)[0]}_dba_plot.png")
                    plt.savefig(dba_plot_path)
                    plt.close()

                    plots.append({
                        "File": os.path.relpath(file_path, data_folder),
                        "dB(A) Plot Path": dba_plot_path
                    })

                    print(f"Plot saved to {dba_plot_path}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Save valid ranges to JSON (keep in the root folder)
    json_path = os.path.join(data_folder, "valid_ranges.json")
    with open(json_path, "w") as json_file:
        json.dump(valid_ranges, json_file, indent=4)
    print(f"Valid ranges saved to {json_path}")

    # Calculate total discarded percentage
    total_discarded_percentage = ((total_raw_minutes - total_good_minutes) / total_raw_minutes) * 100 if total_raw_minutes > 0 else 0

    return pd.DataFrame(results), plots, total_raw_minutes, total_good_minutes, total_discarded_percentage, calibration_offsets, calibration_before_plot, calibration_after_plot

def generate_html_report(results_df, plots, output_path, total_raw_minutes, total_good_minutes, total_discarded_percentage, calibration_offsets, calibration_before_plot, calibration_after_plot):
    """
    Generates an HTML report with the results and plots.
    """
    template = Template("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Processing Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            img { max-width: 48%; height: auto; margin-right: 2%; }
        </style>
    </head>
    <body>
        <h1>Data Processing Report</h1>
        <h2>Summary</h2>
        <p><strong>Total Raw Data:</strong> {{ total_raw_minutes }} minutes</p>
        <p><strong>Total Good Data:</strong> {{ total_good_minutes }} minutes</p>
        <p><strong>Percentage of Data Discarded:</strong> {{ total_discarded_percentage }}%</p>
        <h2>Calibration Results</h2>
        <p><strong>Time Offsets:</strong></p>
        <ul>
            {% for file, offset in calibration_offsets %}
            <li>{{ file }}: {{ offset }} seconds</li>
            {% endfor %}
        </ul>
        <h3>Calibration Plots</h3>
        <img src="{{ calibration_before_plot }}" alt="Calibration Before Alignment">
        <img src="{{ calibration_after_plot }}" alt="Calibration After Alignment">
        <h2>Details</h2>
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th>Device Info</th>
                    <th>Average dB(A)</th>
                    <th>Average Valid dB(A)</th>
                    <th>Discarded (%)</th>
                </tr>
            </thead>
            <tbody>
                {% for row in results %}
                <tr>
                    <td>{{ row["File"] }}</td>
                    <td>{{ row["Device Info"] }}</td>
                    <td>{{ row["Average dB(A)"] }}</td>
                    <td>{{ row["Average Valid dB(A)"] }}</td>
                    <td>{{ row["Discarded (%)"] }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        <h2>Plots</h2>
        {% for plot in plots %}
        <h3>{{ plot["File"] }}</h3>
        <img src="{{ plot["dB(A) Plot Path"] }}" alt="dB(A) Plot for {{ plot['File'] }}">
        {% endfor %}
    </body>
    </html>
    """)

    html_content = template.render(
        results=results_df.to_dict(orient="records"),
        plots=plots,
        total_raw_minutes=round(total_raw_minutes, 2),
        total_good_minutes=round(total_good_minutes, 2),
        total_discarded_percentage=round(total_discarded_percentage, 2),
        calibration_offsets=calibration_offsets,
        calibration_before_plot=calibration_before_plot,
        calibration_after_plot=calibration_after_plot
    )
    with open(output_path, "w") as f:
        f.write(html_content)
    print(f"HTML report saved to {output_path}")

if __name__ == "__main__":
    main_folder = "/home/stan/Documents/noise_project"

    data_folder = os.path.join(main_folder, "curated_data")
    output_folder = os.path.join(main_folder, "processed_output")
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(data_folder):
        print(f"Folder '{data_folder}' does not exist.")
        exit()

    results_df, plots, total_raw_minutes, total_good_minutes, total_discarded_percentage = process_folder(data_folder, output_folder)

    if results_df.empty:
        print("No valid recordings found.")
    else:
        # Save the HTML report in the processed_output folder
        html_path = os.path.join(output_folder, "lday_report.html")
        generate_html_report(results_df, plots, html_path, total_raw_minutes, total_good_minutes, total_discarded_percentage)