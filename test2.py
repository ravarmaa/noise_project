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


import os
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment

def dbfs(samples):
    rms = np.sqrt(np.mean(samples ** 2))
    return 20 * np.log10(rms + 1e-12)

def process_calibration_files(folder_path, output_folder):
    """
    Load and plot raw dBFS signals from calibration recordings.
    No calibration or dBA conversion is applied.
    """
    calibration_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower() in {"calibration.m4a", "calibration.mp4"}:
                calibration_files.append(os.path.join(root, f))

    if not calibration_files:
        print("No calibration files found.")
        return None

    signals = []  # (device_name, dbfs_levels, timestamps)

    for file_path in calibration_files:
        audio = AudioSegment.from_file(file_path)
        fs = audio.frame_rate
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        bit_depth = audio.sample_width * 8
        samples /= float(2 ** (bit_depth - 1))
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels)).mean(axis=1)

        interval = fs
        dbfs_levels = []
        timestamps = []

        for i in range(0, len(samples) - interval + 1, interval):
            chunk = samples[i:i + interval]
            dbfs_levels.append(dbfs(chunk))
            timestamps.append(i / fs)

        device_name = os.path.basename(os.path.dirname(file_path))
        signals.append((device_name, dbfs_levels, timestamps))

    # Plot raw dBFS signals

    plot_path = os.path.join(output_folder, "calibration_dbfs_plot.png")
    plt.figure(figsize=(12, 6))
    for device_name, dbfs_levels, timestamps in signals:
        plt.plot(timestamps, dbfs_levels, label=device_name)
    plt.legend()
    plt.title("Raw dBFS Signals from Calibration Recordings")
    plt.xlabel("Time (s)")
    plt.ylabel("dBFS")
    plt.grid()
    plt.savefig(plot_path)
    plt.close()

    print(f"Raw dBFS plot saved to {plot_path}")
    return plot_path


def process_file(file_path, calibration_offset=132.53):
    audio = AudioSegment.from_file(file_path)
    duration = len(audio)  # in milliseconds
    step = 1000
    timestamps = []
    dba_levels = []
    dbfs_levels = []

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
        dbfs_level = dbfs(samples[start:end])  # Raw dBFS levels
        dbfs_levels.append(dbfs_level)
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

    return dbfs_levels, fs, timestamps, dba_levels, valid_timestamps, valid_dba_levels, discarded_percentage


def process_folder(data_folder, output_folder):
    results = []
    plots = []
    valid_ranges = {}
    total_raw_minutes = 0
    total_good_minutes = 0

    # Process calibration files
    calibration_plot = process_calibration_files(data_folder, output_folder)

    for root, dirs, files in os.walk(data_folder):
        relative_path = os.path.relpath(root, data_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

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
                    dbfs_levels, fs, timestamps, dba_levels, valid_timestamps, valid_dba_levels, discarded_percentage = process_file(file_path, calibration_offset=132.53)

                    avg_dba = np.mean(dba_levels)
                    avg_valid_dba = np.mean(valid_dba_levels)

                    raw_minutes = len(timestamps) / 60
                    good_minutes = len(valid_timestamps) / 60
                    total_raw_minutes += raw_minutes
                    total_good_minutes += good_minutes

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

                    # Plot raw dBFS levels
                    plt.figure(figsize=(12, 8))
                    plt.plot(timestamps, dbfs_levels, label="Raw dBFS Levels")
                    plt.xlabel("Time (s)")
                    plt.ylabel("dBFS")
                    plt.title(f"Raw dBFS Levels for {os.path.basename(file_path)}")
                    plt.legend()
                    plt.grid()
                    raw_plot_path = os.path.join(output_subfolder, f"{os.path.splitext(f)[0]}_raw_plot.png")
                    plt.savefig(raw_plot_path)
                    plt.close()

                    # Plot cropped dBA levels
                    plt.figure(figsize=(12, 8))
                    plt.plot(valid_timestamps, valid_dba_levels, label="Cropped dBA Levels")
                    plt.xlabel("Time (s)")
                    plt.ylabel("dB(A)")
                    plt.title(f"Cropped dBA Levels for {os.path.basename(file_path)}")
                    plt.legend()
                    plt.grid()
                    dba_plot_path = os.path.join(output_subfolder, f"{os.path.splitext(f)[0]}_dba_plot.png")
                    plt.savefig(dba_plot_path)
                    plt.close()

                    plots.append({
                        "File": os.path.relpath(file_path, data_folder),
                        "Raw Plot Path": raw_plot_path,
                        "dB(A) Plot Path": dba_plot_path
                    })

                    print(f"Plots saved to {raw_plot_path} and {dba_plot_path}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    json_path = os.path.join(data_folder, "valid_ranges.json")
    with open(json_path, "w") as json_file:
        json.dump(valid_ranges, json_file, indent=4)
    print(f"Valid ranges saved to {json_path}")

    total_discarded_percentage = ((total_raw_minutes - total_good_minutes) / total_raw_minutes) * 100 if total_raw_minutes > 0 else 0

    return pd.DataFrame(results), plots, total_raw_minutes, total_good_minutes, total_discarded_percentage, calibration_plot

def generate_html_report(results_df, plots, output_path, total_raw_minutes, total_good_minutes, total_discarded_percentage, calibration_plot):
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
            .plot-container { display: flex; justify-content: space-between; margin-bottom: 40px; }
        </style>
    </head>
    <body>
        <h1>Data Processing Report</h1>
        <h2>Summary</h2>
        <p><strong>Total Raw Data:</strong> {{ total_raw_minutes }} minutes</p>
        <p><strong>Total Good Data:</strong> {{ total_good_minutes }} minutes</p>
        <p><strong>Percentage of Data Discarded:</strong> {{ total_discarded_percentage }}%</p>
        <h2>Calibration Plot</h2>
        <img src="{{ calibration_plot }}" alt="Calibration Plot">
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
        <div class="plot-container">
            <img src="{{ plot["Raw Plot Path"] }}" alt="Raw dBFS Plot for {{ plot['File'] }}">
            <img src="{{ plot["dB(A) Plot Path"] }}" alt="Cropped dBA Plot for {{ plot['File'] }}">
        </div>
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
        calibration_plot=calibration_plot
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

    results_df, plots, total_raw_minutes, total_good_minutes, total_discarded_percentage, calibration_plot = process_folder(data_folder, output_folder)

    if results_df.empty:
        print("No valid recordings found.")
    else:
        # Save the HTML report in the processed_output folder
        html_path = os.path.join(output_folder, "processing_report.html")
        generate_html_report(results_df, plots, html_path, total_raw_minutes, total_good_minutes, total_discarded_percentage, calibration_plot)