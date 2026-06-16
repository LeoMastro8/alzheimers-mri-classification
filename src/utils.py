from pathlib import Path
import config


def clean_subject_name(name: str) -> str:
    """Convert folder/file naming to split-file naming style."""
    # folder example: 002_S_0413_init  -> split style: 002S0413init
    return name.replace("_", "")


def list_real_files(folder: Path):
    """Return only real files, ignoring macOS metadata files."""
    return [f for f in folder.iterdir() if f.is_file() and not f.name.startswith("._")]


def detect_modalities(subject_dir: Path):
    """Detect which modalities exist for one subject folder."""
    files = list_real_files(subject_dir)
    names = [f.name for f in files]

    found = {}
    for mod in config.MODALITIES:
        nii_exists = any(name.endswith(f"_{mod}.nii.gz") for name in names)
        json_exists = any(name.endswith(f"_{mod}.json") for name in names)
        found[mod] = {"nii.gz": nii_exists, "json": json_exists}
    return found
