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


import json
from scipy.signal import find_peaks

def extract_above_dba_segments(file_path, timestamps, dba_levels, output_folder, threshold=60):
    """
    Extracts segments where the calibrated dBA level exceeds the given threshold and saves them to the output folder.
    If the total number of events is less than 10, selects the top 10 peaks regardless of the threshold.
    Adds a 3-second margin before and after each event.
    Outputs a JSON file with event metadata.
    If the JSON file already exists, skips extraction and removes unlisted MP3 files.
    """
    # Create the output folder if it doesn't exist
    above_folder = os.path.join(output_folder, f"above_{threshold}_dba")

    print(above_folder)
    os.makedirs(above_folder, exist_ok=True)

    # Define the JSON path
    json_path = os.path.join(above_folder, f"{os.path.splitext(os.path.basename(file_path))[0]}_events.json")

    # If the JSON file already exists, skip extraction
    if os.path.exists(json_path):
        print(f"JSON file already exists: {json_path}. Skipping extraction.")
        # Load the existing JSON file
        with open(json_path, "r") as json_file:
            events = json.load(json_file)

        # Get the list of filenames from the JSON
        valid_filenames = {event["filename"] for event in events}

        # Remove any MP3 files in the directory that are not in the JSON
        for file in os.listdir(above_folder):
            if file.endswith(".mp3") and file not in valid_filenames:
                file_to_remove = os.path.join(above_folder, file)
                os.remove(file_to_remove)
                print(f"Removed unlisted file: {file_to_remove}")

        return  # Exit the function

    # Load the original audio file
    audio = AudioSegment.from_file(file_path)

    # Identify segments where dBA > threshold
    margin = 3  # seconds
    segments = []

    # Identify segments where dBA > threshold
    start_idx = None
    for i, dba in enumerate(dba_levels):
        if dba > threshold:
            if start_idx is None:
                start_idx = i
        else:
            if start_idx is not None:
                # End of a segment
                end_idx = i - 1
                segments.append((start_idx, end_idx))
                start_idx = None

    # Handle the case where the last segment extends to the end
    if start_idx is not None:
        segments.append((start_idx, len(dba_levels) - 1))

    # If the number of segments is less than 10, add unique peaks to reach 10 total
    if len(segments) < 10:
        # Find peaks with a minimum distance between them
        peak_indices, _ = find_peaks(dba_levels, distance=5)  # Minimum distance of 5 indices between peaks
        sorted_peaks = sorted(peak_indices, key=lambda i: dba_levels[i], reverse=True)  # Sort peaks by dBA level

        # Track indices already covered by existing segments
        covered_indices = set()
        for start, end in segments:
            covered_indices.update(range(start, end + 1))

        # Add unique peaks not already covered by segments
        for peak_idx in sorted_peaks:
            if peak_idx not in covered_indices:
                segments.append((peak_idx, peak_idx))  # Treat the peak as a single-point segment
                covered_indices.add(peak_idx)
            if len(segments) >= 10:
                break

    # Prepare JSON metadata
    events = []

    # Extract and save each segment
    for start_idx, end_idx in segments:
        # Calculate start and end times with margin
        start_time = max(0, timestamps[start_idx] - margin) * 1000  # Convert to milliseconds
        end_time = min(timestamps[end_idx] + margin, len(audio) / 1000) * 1000  # Convert to milliseconds

        # Extract the segment
        segment = audio[start_time:end_time]

        # Get the peak dB level for the segment
        peak_dba = max(dba_levels[start_idx:end_idx + 1])

        # Generate a filename with the timestamp and peak dB level of the event
        event_time = timestamps[start_idx]
        filename = f"{os.path.splitext(os.path.basename(file_path))[0]}_above_{threshold}_{event_time:.2f}s_{peak_dba:.2f}dB.mp3"
        output_path = os.path.join(above_folder, filename)

        # Save the segment
        segment.export(output_path, format="mp3")
        print(f"Saved segment: {output_path}")

        # Add event metadata
        events.append({
            "start_time": start_time / 1000,  # Convert back to seconds
            "end_time": end_time / 1000,  # Convert back to seconds
            "filename": filename,
            "peak_dba": peak_dba,
            "valid": True  # Default to valid
        })

    # Save events to JSON
    with open(json_path, "w") as json_file:
        json.dump(events, json_file, indent=4)
    print(f"Saved events JSON: {json_path}")

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
    Load and plot A-weighted dBFS signals from calibration recordings.
    Synchronize signals using least squares fitting for both time and y-scale offsets.
    """
    calibration_files = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.lower() in {"calibration.m4a", "calibration.mp4"}:
                calibration_files.append(os.path.join(root, f))

    if not calibration_files:
        print("No calibration files found.")
        return None, None, None, []

    signals = []  # (device_name, a_weighted_levels, timestamps)
    offsets = []  # Store time and y-scale offsets for each device

    for file_path in calibration_files:
        audio = AudioSegment.from_file(file_path)
        fs = audio.frame_rate
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        bit_depth = audio.sample_width * 8
        samples /= float(2 ** (bit_depth - 1))
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels)).mean(axis=1)

        interval = fs
        a_weighted_levels = []
        timestamps = []
        weighted_samples = apply_a_weighting(samples, fs)

        for i in range(0, len(samples) - interval + 1, interval):
            chunk = weighted_samples[i:i + interval]
            a_weighted_levels.append(dbfs(chunk))
            timestamps.append(i / fs)

        device_name = os.path.basename(os.path.dirname(file_path))
        signals.append((device_name, a_weighted_levels, timestamps))

    # Find the reference signal (contains "Rando")
    reference_signal = None
    for device_name, a_weighted_levels, timestamps in signals:
        if "Rando" in device_name:
            reference_signal = (device_name, a_weighted_levels, timestamps)
            break

    if reference_signal is None:
        raise ValueError("No calibration file with 'Rando' found.")

    ref_device_name, ref_levels, ref_timestamps = reference_signal

    # Synchronize signals using fit_time_and_y_offset
    synced_signals = [(ref_device_name, ref_levels, ref_timestamps)]  # Add the reference signal

    for device_name, levels, timestamps in signals:
        if device_name == ref_device_name:
            continue

        time_offset, y_offset = fit_time_and_y_offset(ref_levels, ref_timestamps, levels, timestamps)
        offsets.append({"Device": device_name, "Time Offset (s)": round(time_offset, 3), "Y Offset (dB)": round(y_offset, 3)})

        # Apply the offsets
        shifted_timestamps = [t + time_offset for t in timestamps]
        interp_func = interp1d(shifted_timestamps, levels, bounds_error=False, fill_value="extrapolate")
        aligned_levels = interp_func(ref_timestamps) + y_offset
        synced_signals.append((device_name, aligned_levels, ref_timestamps))

    # Plot synced A-weighted dBFS signals
    synced_plot_path = os.path.join(output_folder, "calibration_a_weighted_synced_plot.png")
    plt.figure(figsize=(12, 6))
    for device_name, levels, timestamps in synced_signals:
        plt.plot(timestamps, levels, label=device_name)
    plt.legend()
    plt.title("A-Weighted dBFS Signals from Calibration Recordings (Synced)")
    plt.xlabel("Time (s)")
    plt.ylabel("A-Weighted dBFS")
    plt.grid()
    plt.savefig(synced_plot_path)
    plt.close()

    # Plot unsynced A-weighted dBFS signals
    unsynced_plot_path = os.path.join(output_folder, "calibration_a_weighted_unsynced_plot.png")
    plt.figure(figsize=(12, 6))
    for device_name, levels, timestamps in signals:
        plt.plot(timestamps, levels, label=device_name)
    plt.legend()
    plt.title("A-Weighted dBFS Signals from Calibration Recordings (Unsynced)")
    plt.xlabel("Time (s)")
    plt.ylabel("A-Weighted dBFS")
    plt.grid()
    plt.savefig(unsynced_plot_path)
    plt.close()

    print(f"Unsynced plot saved to {unsynced_plot_path}")
    print(f"Synced plot saved to {synced_plot_path}")
    
    return (unsynced_plot_path, synced_plot_path), offsets


def process_file(file_path, calibration_offset=132.53, threshold=60):
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

    json_path = os.path.join(
        os.path.dirname(file_path),
        f"above_{threshold}_dba",
        f"{os.path.splitext(os.path.basename(file_path))[0]}_events.json"
    )

    invalid_ranges = []
    if os.path.exists(json_path):
        with open(json_path, "r") as json_file:
            events = json.load(json_file)
            invalid_ranges = [(event["start_time"], event["end_time"]) for event in events if not event["valid"]]

    for t in range(valid_start, valid_end, step):
        start = int(t * fs / 1000)
        end = int((t + step) * fs / 1000)
        timestamp = t / 1000

        # Check if the timestamp falls within any invalid range
        if any(start <= timestamp <= end for start, end in invalid_ranges):
            continue

        valid_dba_levels.append(dbfs(weighted[start:end]) + calibration_offset)
        valid_timestamps.append(timestamp)

    total_duration = duration / 1000  # Convert milliseconds to seconds

    # Calculate the valid duration based on valid_start and valid_end
    valid_duration = (valid_end - valid_start) / 1000  # Convert milliseconds to seconds

    # Calculate the invalid duration from invalid_ranges
    invalid_duration = 0
    for start, end in invalid_ranges:
        # Only consider ranges that overlap with the valid range
        overlap_start = max(start, valid_start / 1000)  # Convert valid_start to seconds
        overlap_end = min(end, valid_end / 1000)  # Convert valid_end to seconds
        if overlap_start < overlap_end:  # Ensure there is an overlap
            invalid_duration += overlap_end - overlap_start

    # Calculate the final valid duration
    final_valid_duration = valid_duration - invalid_duration

    # Calculate the discarded duration
    discarded_duration = total_duration - final_valid_duration

    # Calculate the discarded percentage
    discarded_percentage = (discarded_duration / total_duration) * 100 if total_duration > 0 else 0

    # Calculate Lday
    if valid_dba_levels:
        lday = 10 * np.log10(np.mean(10 ** (np.array(valid_dba_levels) / 10)))
    else:
        lday = None

    return dbfs_levels, fs, timestamps, dba_levels, valid_timestamps, valid_dba_levels, discarded_percentage, lday


from scipy.interpolate import interp1d
from scipy.optimize import least_squares
import numpy as np

def fit_time_and_y_offset(reference_levels, reference_timestamps, levels, timestamps):
    """
    Fits both time and y-scale offsets between a reference signal and another signal.
    Returns the time offset (in seconds) and y-scale offset (in dB).
    """
    def residuals(params):
        time_offset, y_offset = params
        shifted_timestamps = [t + time_offset for t in timestamps]
        interp_func = interp1d(shifted_timestamps, levels, bounds_error=False, fill_value="extrapolate")
        aligned_levels = interp_func(reference_timestamps) + y_offset
        return reference_levels - aligned_levels

    # Initial guess: no time offset, no y-scale offset
    initial_guess = [0.0, 0.0]
    result = least_squares(residuals, initial_guess, bounds=([-10.0, -20.0], [10.0, 20.0]), loss='soft_l1', f_scale=0.1)
    return result.x[0], result.x[1]  # time_offset, y_offset

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
    calibration_plot, offsets = process_calibration_files(data_folder, output_folder)
    y_offsets = {offset["Device"]: offset["Y Offset (dB)"] for offset in offsets}

    for root, dirs, files in os.walk(data_folder):
        relative_path = os.path.relpath(root, data_folder)
        output_subfolder = os.path.join(output_folder, relative_path)

        if relative_path == "." or 'above_60_dba' in relative_path.lower():
            continue

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


        # elif 'rando' not in relative_path.lower():
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

                device_name = os.path.basename(os.path.dirname(file_path))
                y_offset = y_offsets.get(device_name, 0.0)
            

                dbfs_levels, fs, timestamps, dba_levels, valid_timestamps, valid_dba_levels, discarded_percentage, lday = process_file(
                    file_path, calibration_offset=132.53 + y_offset)
                
                extract_above_dba_segments(file_path, valid_timestamps, valid_dba_levels, root)

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
                    "Calibration Offset Compared to Rando (dB)": round(y_offset, 2),
                    "Average indoor dB(A)": round(avg_valid_dba, 2),
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
                    <th>Calibration Offset Compared to Rando (dB)</th>
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
                    <td>{{ row["Calibration Offset Compared to Rando (dB)"] }}</td>
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