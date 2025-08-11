import os
import shutil
import subprocess
import tempfile

DATASETS_DIR = "datasets"


def download_multi3nlu():
    name = "nlupp_english"
    target_path = os.path.join("datasets", name)
    repo_url = "https://github.com/PolyAI-LDN/task-specific-datasets.git"
    folder_in_repo = "nlupp/data"

    if os.path.exists(target_path):
        print(f"'{name}' directory already exists. Skipping.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, temp_dir],
            check=True,
            capture_output=True,
        )

        source_folder = os.path.join(temp_dir, folder_in_repo)

        shutil.move(source_folder, target_path)


def download_blendx_datasets():
    repo_url = "https://github.com/HYU-NLP/BlendX"
    datasets = {
        "banking77": "Banking77",
        "clinc150": "CLINC150"
    }

    banking_path = os.path.join("datasets", "banking77")
    clinc_path = os.path.join("datasets", "clinc150")

    if os.path.exists(banking_path) and os.path.exists(clinc_path):
        print("'banking77' and 'clinc150' directories already exist. Skipping.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, temp_dir],
            check=True,
            capture_output=True
        )        
        
        for name, original_name in datasets.items():
            target_dir = os.path.join("datasets", name)
            os.makedirs(target_dir, exist_ok=True)

            source_blend_file = os.path.join(temp_dir, "v1.0", "BlendX", f"Blend{original_name}.json")

            dest_blend_file = os.path.join(target_dir, "blend.json")

            shutil.move(source_blend_file, dest_blend_file)





if __name__ == "__main__":
    os.makedirs(DATASETS_DIR, exist_ok=True)

    # download_multi3nlu()
    # download_blendx_datasets()

