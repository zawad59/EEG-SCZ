import cupy as cp
import numpy as np
import mne
from tqdm import tqdm
import os
from datetime import datetime
import matplotlib.pyplot as plt


def create_morlet_kernel_gpu(freq, sfreq, n_cycles=7):
    """Create Morlet wavelet on GPU"""
    st = n_cycles / (2 * np.pi * freq)  # Standard deviation in time
    t = cp.arange(-3.5 * st, 3.5 * st, 1 / sfreq, dtype=cp.float32)
    wavelet = cp.exp(2 * np.pi * 1j * freq * t) * cp.exp(-t**2 / (2 * st**2))
    return wavelet / cp.sqrt(cp.sum(cp.abs(wavelet)**2))


def morlet_convolution_gpu(signal_gpu, wavelet_gpu, n_orig):
    """GPU-accelerated convolution via FFT"""
    m = len(wavelet_gpu)
    N = n_orig + m - 1
    signal_fft = cp.fft.fft(signal_gpu, N)
    kernel_fft = cp.fft.fft(wavelet_gpu, N)
    conv = cp.fft.ifft(signal_fft * kernel_fft)[:n_orig + m - 1]
    start = m // 2
    return conv[start:start + n_orig]


def compute_tfr_gpu(raw, freqs=np.arange(1, 51, 1), n_cycles=7):
    """Compute TFR for all channels on GPU"""
    sfreq = raw.info['sfreq']
    data = raw.get_data()
    n_channels, n_times = data.shape

    data_gpu = cp.asarray(data.astype(np.float32))
    tf_results = cp.zeros((len(freqs), n_channels, n_times), dtype=cp.complex64)

    for i, freq in enumerate(tqdm(freqs, desc="Processing frequencies")):
        wavelet_gpu = create_morlet_kernel_gpu(freq, sfreq, n_cycles)
        for ch in range(n_channels):
            tf_results[i, ch] = morlet_convolution_gpu(
                data_gpu[ch],
                wavelet_gpu,
                n_times
            )

    return cp.asnumpy(tf_results)


def process_subject_gpu(subject_dir, output_dir="gpu_tfr_results", sfreq_new=100):
    """Optimized GPU TFR pipeline with improved visualization"""
    print(f"\nProcessing {subject_dir} with GPU acceleration")
    start_time = datetime.now()

    vhdr_file = next(f for f in os.listdir(subject_dir) if f.endswith('.vhdr'))
    raw = mne.io.read_raw_brainvision(os.path.join(subject_dir, vhdr_file), preload=True)

    raw.filter(0.5, 50, fir_design='firwin', verbose=False)

    if raw.info['sfreq'] > sfreq_new:
        raw.resample(sfreq_new, verbose=False)

    freqs = np.arange(1, 51, 1)
    tf_results = compute_tfr_gpu(raw, freqs=freqs)

    os.makedirs(output_dir, exist_ok=True)

    power = np.abs(tf_results) ** 2  # raw power
    power_db = 10 * np.log10(power + 1e-20)  # dB scale

    for ch_idx, ch_name in enumerate(raw.ch_names):
        plt.figure(figsize=(12, 8))

        # Baseline correction: subtract mean over time per freq
        baseline = np.mean(power_db[:, ch_idx, :], axis=1, keepdims=True)
        power_rel = power_db[:, ch_idx, :] - baseline

        # Robust color scaling (ignore extreme outliers)
        vmin, vmax = np.percentile(power_rel, [5, 95])

        plt.imshow(power_rel,
                   aspect='auto',
                   origin='lower',
                   extent=[raw.times[0], raw.times[-1], freqs[0], freqs[-1]],
                   cmap='magma',
                   vmin=vmin, vmax=vmax)

        # Add horizontal lines for known frequency bands
        for band in [4, 8, 13, 30]:
            plt.axhline(band, color='white', linestyle='--', linewidth=0.5)

        plt.colorbar(label='Relative Power (dB)')
        plt.ylabel("Frequency (Hz)")
        plt.xlabel("Time (s)")
        plt.title(f"GPU-TFR (relative to mean): {ch_name}")
        plt.savefig(os.path.join(output_dir, f"gpu_tfr_{ch_name}.png"), bbox_inches='tight')
        plt.close()

    print(f"Completed in {datetime.now() - start_time}")


def visualize_tfr_results(subject_dir, output_dir="gpu_tfr_results"):
    """Interactive visualization of all processed channels"""
    from matplotlib.widgets import RadioButtons

    raw = mne.io.read_raw_brainvision(
        os.path.join(subject_dir, [f for f in os.listdir(subject_dir) if f.endswith('.vhdr')][0]),
        preload=False
    )

    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(bottom=0.25)

    result_files = sorted([f for f in os.listdir(output_dir) if f.startswith('gpu_tfr_')])
    channel_names = [f.replace('gpu_tfr_', '').replace('.png', '') for f in result_files]

    img = plt.imread(os.path.join(output_dir, result_files[0]))
    img_plot = ax.imshow(img)
    ax.set_title(f"Channel: {channel_names[0]}")
    ax.axis('off')

    ax_radio = plt.axes([0.15, 0.05, 0.7, 0.1])
    radio = RadioButtons(ax_radio, channel_names)

    def update(label):
        idx = channel_names.index(label)
        img_plot.set_data(plt.imread(os.path.join(output_dir, result_files[idx])))
        ax.set_title(f"Channel: {label}")
        fig.canvas.draw()

    radio.on_clicked(update)
    plt.show()


if __name__ == "__main__":
    subject_directory = 'Z:/EEG Data/CA03409/eeg/ses-20230127/CA03409_eeg_visit001/'
    output_directory = "gpu_tfr_results"

    process_subject_gpu(subject_directory, output_directory)
    visualize_tfr_results(subject_directory, output_directory)
