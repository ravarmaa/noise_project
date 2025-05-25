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

from scipy.optimize import least_squares

def process_calibration_files(folder_path, output_folder):
    """
    Load and plot raw dBFS signals from calibration recordings.
    Synchronize signals using least squares fitting.
    """
    calibration_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower() in {"calibration.m4a", "calibration.mp4"}:
                calibration_files.append(os.path.join(root, f))

    if not calibration_files:
        print("No calibration files found.")
        return None, None

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
    unsynced_plot_path = os.path.join(output_folder, "calibration_dbfs_unsynced_plot.png")
    plt.figure(figsize=(12, 6))
    for device_name, dbfs_levels, timestamps in signals:
        plt.plot(timestamps, dbfs_levels, label=device_name)
    plt.legend()
    plt.title("Raw dBFS Signals from Calibration Recordings (Unsynced)")
    plt.xlabel("Time (s)")
    plt.ylabel("dBFS")
    plt.grid()
    plt.savefig(unsynced_plot_path)
    plt.close()

    # Synchronize signals using calculate_time_offset
    reference_signal = signals[0][1]  # Use the first device as the reference
    reference_timestamps = signals[0][2]

    synced_signals = [(signals[0][0], reference_signal, reference_timestamps)]  # Add the reference signal

    for device_name, dbfs_levels, timestamps in signals[1:]:
        offset = calculate_time_offset(reference_signal, dbfs_levels)
        shifted_timestamps = [t - offset for t in timestamps]
        synced_signals.append((device_name, dbfs_levels, shifted_timestamps))
        print(f"Calculated offset for {device_name}: {offset:.3f} seconds")

    # Plot synced dBFS signals
    synced_plot_path = os.path.join(output_folder, "calibration_dbfs_synced_plot.png")
    plt.figure(figsize=(12, 6))
    for device_name, dbfs_levels, timestamps in synced_signals:
        plt.plot(timestamps, dbfs_levels, label=device_name)
    plt.legend()
    plt.title("Raw dBFS Signals from Calibration Recordings (Synced)")
    plt.xlabel("Time (s)")
    plt.ylabel("dBFS")
    plt.grid()
    plt.savefig(synced_plot_path)
    plt.close()

    print(f"Unsynced plot saved to {unsynced_plot_path}")
    print(f"Synced plot saved to {synced_plot_path}")
    return unsynced_plot_path, synced_plot_path


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

    # Calculate Lday
    if valid_dba_levels:
        lday = 10 * np.log10(np.mean(10 ** (np.array(valid_dba_levels) / 10)))
    else:
        lday = None

    return dbfs_levels, fs, timestamps, dba_levels, valid_timestamps, valid_dba_levels, discarded_percentage, lday


from scipy.interpolate import interp1d
from scipy.optimize import least_squares
import numpy as np

def calculate_time_offset(reference, signal):
    """
    Calculates sub-second time offset between two 1 Hz signals (e.g., dBFS time series)
    using least squares fitting. Returns a float offset (in seconds).
    Positive offset means `signal` lags behind `reference`.
    """

    reference = np.array(reference)
    signal = np.array(signal)

    if len(signal) != len(reference):
        # Truncate to shortest length for alignment
        min_len = min(len(reference), len(signal))
        reference = reference[:min_len]
        signal = signal[:min_len]

    x = np.arange(len(signal))
    signal_interp = interp1d(x, signal, kind="linear", fill_value="extrapolate")

    def error(offset):
        shifted_x = x + offset[0]
        shifted_signal = signal_interp(shifted_x)
        return reference - shifted_signal

    result = least_squares(error, [0.0], bounds=(-10.0, 10.0))
    return result.x[0]  # float offset in seconds

def process_window_file(file_path, calibration_offset=132.53):
    """
    Processes a window file (opened or closed) to calculate calibrated dBA values.
    Clips 5 seconds from the start and end of the recording.
    """
    audio = AudioSegment.from_file(file_path)
    fs = audio.frame_rate
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    bit_depth = audio.sample_width * 8
    samples /= float(2 ** (bit_depth - 1))
    if audio.channels > 1:
        samples = samples.reshape((-1, audio.channels)).mean(axis=1)

    # Clip 5 seconds from the start and end
    start_clip = 5 * fs
    end_clip = len(samples) - 5 * fs
    if end_clip <= start_clip:
        raise ValueError(f"File {file_path} is too short to clip 5 seconds from start and end.")

    clipped_samples = samples[start_clip:end_clip]
    weighted_samples = apply_a_weighting(clipped_samples, fs)

    # Calculate dBA levels
    step = fs  # 1-second intervals
    dba_levels = []
    timestamps = []

    for i in range(0, len(clipped_samples) - step + 1, step):
        chunk = weighted_samples[i:i + step]
        dba_levels.append(dbfs(chunk) + calibration_offset)
        timestamps.append((start_clip + i) / fs)

    return dba_levels, timestamps

import csv
def process_folder(data_folder, output_folder):
    results = []
    plots = []
    attenuation_results = []
    valid_ranges = {}
    total_raw_minutes = 0
    total_good_minutes = 0

    # Process calibration files
    calibration_plot = process_calibration_files(data_folder, output_folder)

    for root, dirs, files in os.walk(data_folder):
        relative_path = os.path.relpath(root, data_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        # Process window attenuation
        window_opened_path = None
        window_closed_path = None

        # Check for both .m4a and .mp4 extensions for window_opened
        for ext in [".m4a", ".mp4"]:
            potential_path = os.path.join(root, f"window_opened{ext}")
            if os.path.exists(potential_path):
                window_opened_path = potential_path
                break

        # Check for both .m4a and .mp4 extensions for window_closed
        for ext in [".m4a", ".mp4"]:
            potential_path = os.path.join(root, f"window_closed{ext}")
            if os.path.exists(potential_path):
                window_closed_path = potential_path
                break

        # Handle missing window_closed file
        if not window_closed_path:
            recording_01_path = None
            for ext in [".m4a", ".mp4"]:
                potential_path = os.path.join(root, f"recording_01{ext}")
                if os.path.exists(potential_path):
                    recording_01_path = potential_path
                    break

            if recording_01_path:
                audio = AudioSegment.from_file(recording_01_path)
                window_closed_audio = audio[500000:600000]  # 500 to 600 seconds
                window_closed_path = os.path.join(output_subfolder, "window_closed.mp4")
                window_closed_audio.export(window_closed_path, format="mp4")
                print(f"Generated missing window_closed file: {window_closed_path}")

        if relative_path == ".":
            continue

        # elif 'friend' not in relative_path.lower():
        #     continue

        if os.path.exists(window_opened_path) and os.path.exists(window_closed_path):
            window_opened_dba, window_opened_timestamps = process_window_file(window_opened_path)
            window_closed_dba, window_closed_timestamps = process_window_file(window_closed_path)

            mean_opened = np.mean(window_opened_dba)
            mean_closed = np.mean(window_closed_dba)
            attenuation = mean_opened - mean_closed

            attenuation_results.append({
                "Folder": relative_path,
                "Window Opened Mean dBA": round(mean_opened, 2),
                "Window Closed Mean dBA": round(mean_closed, 2),
                "Attenuation (dBA)": round(attenuation, 2)
            })

            # Plot window signals
            plt.figure(figsize=(12, 8))
            plt.plot(window_opened_timestamps, window_opened_dba, label="Window Opened", color="blue")
            plt.plot(window_closed_timestamps, window_closed_dba, label="Window Closed", color="red")
            plt.axhline(mean_opened, color="blue", linestyle="--", label="Mean Opened")
            plt.axhline(mean_closed, color="red", linestyle="--", label="Mean Closed")
            plt.xlabel("Time (s)")
            plt.ylabel("dB(A)")
            plt.title(f"Window Signals for {relative_path}")
            plt.legend()
            plt.grid()
            window_plot_path = os.path.join(output_subfolder, "window_signals_plot.png")
            plt.savefig(window_plot_path)
            plt.close()

            # Add only the window signals plot to the plots list
            plots.append({
                "File": f"Window Signals ({relative_path})",
                "Raw Plot Path": window_plot_path,
                "dB(A) Plot Path": None,  # No cropped dBA plot for window signals
                "dB(A) With Attenuation Plot Path": None  # No attenuation plot for window signals
            })

        # Process main recordings
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

                    dbfs_levels, fs, timestamps, dba_levels, valid_timestamps, valid_dba_levels, discarded_percentage, lday = process_file(
                        file_path, calibration_offset=132.53)

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
                        "Total Duration (s)": round(timestamps[-1] - timestamps[0], 2),
                        "Average indoor dB(A)": round(avg_valid_dba),
                        "Min indoor dB(A)": round(np.min(valid_dba_levels), 2),
                        "Max indoor dB(A)": round(np.max(valid_dba_levels), 2),
                        "Discarded (%)": round(discarded_percentage, 2),
                        "Indoor Lday": round(lday, 2) if lday is not None else "N/A"
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

                    plt.figure(figsize=(12, 8))
                    plt.plot(valid_timestamps, valid_dba_levels, label="dBA (Calibration Only)")
                    plt.xlabel("Time (s)")
                    plt.ylabel("dB(A)")
                    plt.title(f"Calibrated dBA Levels for {os.path.basename(file_path)}")
                    plt.legend()
                    plt.grid()
                    dba_plot_path = os.path.join(output_subfolder, f"{os.path.splitext(f)[0]}_dba_plot.png")
                    plt.savefig(dba_plot_path)
                    plt.close()

                    # Plot estimated outdoor levels by applying attenuation
                    estimated_outdoor_dba = [dba + attenuation for dba in valid_dba_levels]
                    plt.figure(figsize=(12, 8))
                    plt.plot(valid_timestamps, estimated_outdoor_dba, label="Estimated Outdoor dBA (With Attenuation)")
                    plt.xlabel("Time (s)")
                    plt.ylabel("dB(A)")
                    plt.title(f"Estimated Outdoor dBA for {os.path.basename(file_path)}")
                    plt.legend()
                    plt.grid()
                    dba_with_attenuation_plot_path = os.path.join(output_subfolder, f"{os.path.splitext(f)[0]}_dba_with_attenuation_plot.png")
                    plt.savefig(dba_with_attenuation_plot_path)
                    plt.close()

                    # Add all three plots for main recordings
                    plots.append({
                        "File": os.path.relpath(file_path, data_folder),
                        "Raw Plot Path": raw_plot_path,
                        "dB(A) Plot Path": dba_plot_path,
                        "dB(A) With Attenuation Plot Path": dba_with_attenuation_plot_path
                    })

                    # Generate CSV files
                    base_filename = os.path.splitext(f)[0]
                    closed_csv_path = os.path.join(output_subfolder, f"{base_filename}_closed.csv")
                    open_csv_path = os.path.join(output_subfolder, f"{base_filename}_open.csv")

                    # Write closed CSV (calibration offset only)
                    with open(closed_csv_path, mode="w", newline="") as closed_csv:
                        writer = csv.writer(closed_csv)
                        writer.writerow(["Timestamp (s)", "dBA (Calibration Only)"])
                        for ts, dba in zip(valid_timestamps, valid_dba_levels):
                            writer.writerow([ts, dba])

                    # Write open CSV (estimated outdoor level)
                    with open(open_csv_path, mode="w", newline="") as open_csv:
                        writer = csv.writer(open_csv)
                        writer.writerow(["Timestamp (s)", "Estimated Outdoor dBA (With Attenuation)"])
                        for ts, dba in zip(valid_timestamps, estimated_outdoor_dba):
                            writer.writerow([ts, dba])

                    print(f"Plots and CSV files saved for {file_path}")

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    json_path = os.path.join(data_folder, "valid_ranges.json")
    with open(json_path, "w") as json_file:
        json.dump(valid_ranges, json_file, indent=4)
    print(f"Valid ranges saved to {json_path}")

    total_discarded_percentage = ((total_raw_minutes - total_good_minutes) / total_raw_minutes) * 100 if total_raw_minutes > 0 else 0

    return pd.DataFrame(results), plots, total_raw_minutes, total_good_minutes, total_discarded_percentage, calibration_plot, attenuation_results

def generate_html_report(results_df, plots, output_path, total_raw_minutes, total_good_minutes, total_discarded_percentage, calibration_plot, attenuation_results):
    """
    Generates an HTML report with the results and plots.
    """
    unsynced_plot, synced_plot = calibration_plot  # Unpack the unsynced and synced plot paths

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
            img { 
                max-width: 100%;  /* Ensure images fit within the container */
                height: auto;    /* Maintain aspect ratio */
                display: block;  /* Prevent inline spacing issues */
                margin: 0 auto;  /* Center the images */
            }
            .plot-table {
                width: 100%;
                margin-bottom: 40px;
                page-break-inside: avoid; /* Prevent splitting across pages */
            }
            .plot-table td {
                width: 50%; /* Ensure two plots fit side by side */
                text-align: center; /* Center the plots */
                vertical-align: top;
            }
        </style>
    </head>
    <body>
        <h1>Data Processing Report</h1>
        <h2>Summary</h2>
        <p><strong>Total Raw Data:</strong> {{ total_raw_minutes }} minutes</p>
        <p><strong>Total Good Data:</strong> {{ total_good_minutes }} minutes</p>
        <p><strong>Percentage of Data Discarded:</strong> {{ total_discarded_percentage }}%</p>
        <h2>Calibration Plots</h2>
        <table class="plot-table">
            <tr>
                <td><img src="{{ unsynced_plot }}" alt="Unsynced Calibration Plot"></td>
                <td><img src="{{ synced_plot }}" alt="Synced Calibration Plot"></td>
            </tr>
        </table>
        <h2>Window Attenuation Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Folder</th>
                    <th>Window Opened Mean dBA</th>
                    <th>Window Closed Mean dBA</th>
                    <th>Attenuation (dBA)</th>
                </tr>
            </thead>
            <tbody>
                {% for row in attenuation_results %}
                <tr>
                    <td>{{ row["Folder"] }}</td>
                    <td>{{ row["Window Opened Mean dBA"] }}</td>
                    <td>{{ row["Window Closed Mean dBA"] }}</td>
                    <td>{{ row["Attenuation (dBA)"] }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        <h2>Details</h2>
        <table>
            <thead>
                <tr>
                    <th>File</th>
                    <th>Device Info</th>
                    <th>Total Duration (s)</th>
                    <th>Discarded (%)</th>
                    <th>Average indoor dB(A)</th>
                    <th>Min indoor dB(A)</th>
                    <th>Max indoor dB(A)</th>
                    <th>Indoor Lday</th>
                </tr>
            </thead>
            <tbody>
                {% for row in results %}
                <tr>
                    <td>{{ row["File"] }}</td>
                    <td>{{ row["Device Info"] }}</td>
                    <td>{{ row["Total Duration (s)"] }}</td>
                    <td>{{ row["Discarded (%)"] }}</td>
                    <td>{{ row["Average indoor dB(A)"] }}</td>
                    <td>{{ row["Min indoor dB(A)"] }}</td>
                    <td>{{ row["Max indoor dB(A)"] }}</td>
                    <td>{{ row["Indoor Lday"] }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        <h2>Plots</h2>
        {% for plot in plots %}
        <h3>{{ plot["File"] }}</h3>
        <table class="plot-table">
            <tr>
                {% if plot["Raw Plot Path"] %}
                <td><img src="{{ plot["Raw Plot Path"] }}" alt="Raw dBFS Plot for {{ plot['File'] }}"></td>
                {% endif %}
                {% if plot["dB(A) Plot Path"] %}
                <td><img src="{{ plot["dB(A) Plot Path"] }}" alt="Cropped dBA Plot for {{ plot['File'] }}"></td>
                {% endif %}
            </tr>
            <tr>
                {% if plot["dB(A) With Attenuation Plot Path"] %}
                <td colspan="2"><img src="{{ plot["dB(A) With Attenuation Plot Path"] }}" alt="Cropped dBA Plot (With Window Attenuation) for {{ plot['File'] }}"></td>
                {% endif %}
            </tr>
        </table>
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
        unsynced_plot=unsynced_plot,
        synced_plot=synced_plot,
        attenuation_results=attenuation_results
    )
    with open(output_path, "w") as f:
        f.write(html_content)
    print(f"HTML report saved to {output_path}")


import pdfkit

def generate_pdf_report(html_path, pdf_path):
    """
    Converts an HTML report to a PDF report using pdfkit.
    """
    options = {
        'enable-local-file-access': None  # Allow access to local files
    }
    try:
        pdfkit.from_file(html_path, pdf_path, options=options)
        print(f"PDF report saved to {pdf_path}")
    except Exception as e:
        print(f"Error generating PDF report: {e}")

if __name__ == "__main__":
    main_folder = "/home/stan/Documents/noise_project"

    data_folder = os.path.join(main_folder, "curated_data")
    output_folder = os.path.join(main_folder, "processed_output")
    os.makedirs(output_folder, exist_ok=True)

    if not os.path.exists(data_folder):
        print(f"Folder '{data_folder}' does not exist.")
        exit()

    results_df, plots, total_raw_minutes, total_good_minutes, total_discarded_percentage, calibration_plot, attenuation_results = process_folder(data_folder, output_folder)

    if results_df.empty:
        print("No valid recordings found.")
    else:
        # Save the HTML report in the processed_output folder
        html_path = os.path.join(output_folder, "processing_report.html")
        generate_html_report(results_df, plots, html_path, total_raw_minutes, total_good_minutes, total_discarded_percentage, calibration_plot, attenuation_results)

        # Convert the HTML report to a PDF report
        pdf_path = os.path.join(output_folder, "processing_report.pdf")
        generate_pdf_report(html_path, pdf_path)