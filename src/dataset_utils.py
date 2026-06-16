import matplotlib.pyplot as plt
import pandas as pd
from ipywidgets import interact


# ---------- data helpers ----------

def sample_indices(dataset, n=5):
    # Use at most n samples, without going out of range.
    return list(range(min(n, len(dataset))))


def sample_sessions(dataset, n=5):
    # Return the session IDs that are being tested.
    return [dataset.records.iloc[i]["session_id"] for i in sample_indices(dataset, n)]


def expected_labels(csv_path, split_name):
    # Build a lookup table: session_id -> label_num.
    df = pd.read_csv(csv_path)
    split_df = df[df["split"] == split_name]
    return dict(zip(split_df["session_id"], split_df["label_num"]))


def load_sample(dataset, idx):
    # Wrap dataset access so the check functions stay short and readable.
    t1, t2, flair, label, session_id = dataset[idx]
    return {
        "session_id": session_id,
        "label": int(label.item()),
        "t1": t1,
        "t2": t2,
        "flair": flair,
    }


# ---------- checks  ----------

def check_shapes(dataset, n=5):
    # Load a few subjects and use the first one as the shape reference.
    samples = [load_sample(dataset, i) for i in sample_indices(dataset, n)]
    reference = tuple(sample[mod].shape for mod in (
        "t1", "t2", "flair") for sample in samples[:1])

    # All tested subjects must match the same T1, T2 and FLAIR shapes.
    return all(tuple(sample[mod].shape for mod in ("t1", "t2", "flair")) == reference for sample in samples)


def check_modalities_load(dataset, n=5):
    # If one modality is missing or broken, dataset[idx] will raise an error.
    try:
        for i in sample_indices(dataset, n):
            load_sample(dataset, i)
        return True
    except Exception:
        return False


def check_labels(dataset, csv_path, split_name, n=5):
    # Compare labels returned by the dataset with labels stored in the CSV.
    labels = expected_labels(csv_path, split_name)

    for i in sample_indices(dataset, n):
        sample = load_sample(dataset, i)
        if sample["label"] != labels[sample["session_id"]]:
            return False

    return True


def run_split_checks(dataset, csv_path, split_name, n=5):
    # Group the required checks for one split in a single dictionary.
    return {
        "shapes_consistent": check_shapes(dataset, n),
        "modalities_load": check_modalities_load(dataset, n),
        "labels_match_csv": check_labels(dataset, csv_path, split_name, n),
        "sessions_tested": sample_sessions(dataset, n),
    }


# ---------- visualization ----------

def pick_axial_slice(volume):
    # Volume shape is (1, D, H, W).
    volume = volume.squeeze(0).numpy()  # drop the channel → shape (D, H, W)
    z = volume.shape[2] // 2  # pick the middle index along W axis
    return volume[:, :, z]  # return a 2D slice of shape (D, H)


def plot_subject_modalities(dataset, idx=0):
    # Load one subject and prepare one axial slice per modality.
    sample = load_sample(dataset, idx)
    session_id = sample["session_id"]

    slices = [
        ("T1", pick_axial_slice(sample["t1"])),
        ("T2", pick_axial_slice(sample["t2"])),
        ("FLAIR", pick_axial_slice(sample["flair"])),
    ]

    # Plot the three modalities side by side.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, image) in zip(axes, slices):
        ax.imshow(image.T, cmap="gray", origin="lower")
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(session_id)
    fig.tight_layout()
    plt.show()


def plot_all_slices(volume_tensor, step=10):
    # volume_tensor shape: (1, D, H, W)
    vol = volume_tensor.squeeze(0).numpy()  # (D, H, W)
    slices = [vol[:, :, z] for z in range(0, vol.shape[2], step)]
    n = len(slices)
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 3))
    for ax, slc in zip(axes, slices):
        ax.imshow(slc.T, cmap="gray", origin="lower")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def explore_volume(volume_tensor):
    vol = volume_tensor.squeeze(0).numpy()

    def show(z):
        plt.figure(figsize=(5, 5))
        plt.imshow(vol[:, :, z].T, cmap="gray", origin="lower")
        plt.axis("off")
        plt.title(f"Slice {z}/{vol.shape[2]}")
        plt.show()
    interact(show, z=(0, vol.shape[2] - 1))
