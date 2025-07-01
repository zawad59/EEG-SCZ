import os
import zipfile

# Base directory containing subject folders
base_dir = r"Z:\EEG Data"

# Counter for subjects with at least one successful extraction
successful_subject_extractions = 0

# Traverse each subject folder in the base directory
for subject_name in os.listdir(base_dir):
    subject_path = os.path.join(base_dir, subject_name)

    # Skip non-directory entries
    if not os.path.isdir(subject_path):
        continue

    eeg_root = os.path.join(subject_path, "eeg")
    if not os.path.exists(eeg_root):
        print(f"[⚠️] Skipping {subject_name} — no 'eeg' folder found")
        continue

    extracted_in_this_subject = False  # Flag to track per-subject extraction

    # Search for zip files recursively under eeg/
    for root, _, files in os.walk(eeg_root):
        zip_files = [f for f in files if f.endswith('.zip')]

        if not zip_files:
            continue  # No zips in this subfolder

        for zip_file in zip_files:
            zip_path = os.path.join(root, zip_file)
            extract_dir = os.path.join(root, zip_file.replace(".zip", ""))

            if os.path.exists(extract_dir):
                print(f"[✅] Already extracted: {zip_file}")
                continue

            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                    print(f"[✔️] Extracted: {zip_file} in {root}")
                    extracted_in_this_subject = True
            except Exception as e:
                print(f"[❌] Failed to extract {zip_file} — {e}")

    if extracted_in_this_subject:
        successful_subject_extractions += 1
        print(f"[📊] Total subjects extracted so far: {successful_subject_extractions}")
