import numpy as np
import matplotlib.pyplot as plt
import os

# === Configuration ===
BASE_DIR = r"Z:\EEG Data"  # Update path
PSD_SAVE_PATH = os.path.join(BASE_DIR, "avg_psd.npy")
MAX_FREQ = 500
TARGET_PSD_BINS = 129

def plot_saved_psd(psd_path, max_freq, target_bins):
    if not os.path.exists(psd_path):
        print(f"❌ PSD file not found: {psd_path}")
        return

    avg_psd = np.load(psd_path)
    if avg_psd.shape[0] != target_bins:
        print(f"⚠️ PSD bin count mismatch: Expected {target_bins}, got {avg_psd.shape[0]}")
        return

    freqs = np.linspace(0, max_freq, target_bins)
    plt.figure(figsize=(10, 5))
    plt.semilogy(freqs, avg_psd)
    plt.title("Average EEG Power Spectral Density")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (V²/Hz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_saved_psd(PSD_SAVE_PATH, MAX_FREQ, TARGET_PSD_BINS)
