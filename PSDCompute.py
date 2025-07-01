import os
import numpy as np
import mne
from scipy.signal import welch
import matplotlib.pyplot as plt
import warnings

# === Configuration ===
BASE_DIR = r"Z:\EEG Data"  # Update path as needed
SFREQ = 1000               # Hz
MAX_FREQ = 1000/2             # Nyquist Theorem
TARGET_PSD_BINS = 129      #Randomly assigned based on segment length (nperseg) of Welch method.
PSD_SAVE_PATH = os.path.join(BASE_DIR, "avg_psd.npy")
LOG_FILE = os.path.join(BASE_DIR, "processed_subjects.txt")

warnings.filterwarnings("ignore", category=RuntimeWarning)

def find_vhdr_files(base_dir):
    vhdr_files = []
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".vhdr"):
                vhdr_files.append(os.path.join(root, file))
    return vhdr_files

def load_processed_log():
    return set(open(LOG_FILE).read().splitlines()) if os.path.exists(LOG_FILE) else set()

def append_to_log(vhdr_path):
    with open(LOG_FILE, 'a') as f:
        f.write(vhdr_path + "\n")

def standardize_psd(psd, target_bins):
    if len(psd) == target_bins:
        return psd
    elif len(psd) > target_bins:
        return psd[:target_bins]
    else:
        return np.pad(psd, (0, target_bins - len(psd)), 'constant')

def compute_psd_for_subject(vhdr_path):
    try:
        raw = mne.io.read_raw_brainvision(vhdr_path, preload=True, verbose=False)
        eeg_data = raw.get_data(picks="eeg")
        sfreq = raw.info["sfreq"]

        if int(sfreq) != SFREQ or eeg_data.shape[1] < 1000:
            print(f"⚠️ Skipped {vhdr_path} due to sampling rate or short recording.")
            return None

        nperseg = max(1000, eeg_data.shape[1] // 1000)
        psds = []
        for ch in eeg_data:
            f, pxx = welch(ch, fs=sfreq, nperseg=nperseg)
            f_mask = f <= MAX_FREQ
            psds.append(standardize_psd(pxx[f_mask], TARGET_PSD_BINS))

        return np.mean(psds, axis=0)
    except Exception as e:
        print(f"[❌] Error processing {vhdr_path}: {e}")
        return None

def main():
    vhdr_files = find_vhdr_files(BASE_DIR)
    print(f"🔍 Found {len(vhdr_files)} .vhdr files.")
    processed = load_processed_log()

    avg_psd = np.load(PSD_SAVE_PATH) if os.path.exists(PSD_SAVE_PATH) else None
    count = len(processed)
    print(f"🔁 Loaded {count} already processed subjects.")

    for i, vhdr_path in enumerate(vhdr_files, 1):
        if vhdr_path in processed:
            print(f"⏭️  Skipping already processed: {vhdr_path}")
            continue

        print(f"🧠 Processing ({i}/{len(vhdr_files)}): {vhdr_path}")
        psd = compute_psd_for_subject(vhdr_path)

        if psd is not None:
            avg_psd = psd if avg_psd is None else (avg_psd * count + psd) / (count + 1)
            count += 1
            np.save(PSD_SAVE_PATH, avg_psd)
            append_to_log(vhdr_path)
            print(f"✅ Saved PSD for subject {count} → {os.path.basename(vhdr_path)}")
        else:
            print(f"⚠️ Skipping subject {i} due to PSD issue.")

    if avg_psd is not None:
        freqs = np.linspace(0, MAX_FREQ, TARGET_PSD_BINS)
        plt.figure(figsize=(10, 5))
        plt.semilogy(freqs, avg_psd)
        plt.title("Average EEG Power Spectral Density")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power Spectral Density (V²/Hz)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ No valid PSDs computed.")

if __name__ == "__main__":
    main()
