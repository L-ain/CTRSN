"""Data loading and epoch segmentation for the CTRSN experiments."""

from pathlib import Path

import numpy as np


EEG_CHANNEL_COUNT = 11
EVENT_ROW = 11
WAKE_EVENT = 0
MICROSLEEP_EVENT = 2


def _segment_bounds(event_values, event_code):
    """Return contiguous event segments as [start, end) ranges."""
    positions = np.flatnonzero(event_values == event_code)
    if positions.size == 0:
        return []

    segments = []
    segment_start = int(positions[0])
    for index, position in enumerate(positions):
        position = int(position)
        is_last = index == len(positions) - 1
        if not is_last and position != int(positions[index + 1]) - 1:
            segments.append((segment_start, position))
            segment_start = int(positions[index + 1])
        elif is_last:
            segments.append((segment_start, position))
    return segments


def load_subjects(data_dir="raw/data"):
    """Load recordings that contain at least one Type B microsleep segment.

    Each ``.npy`` file is expected to contain the 11 EEG channels in rows
    0--10 and an event-code row at index 11. Event code 0 denotes
    wakefulness and event code 2 denotes Type B microsleep. Other event
    codes are excluded from the binary CTRSN dataset.
    """
    files = sorted(Path(data_dir).glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {data_dir}")

    recordings = []
    for path in files:
        recording = np.load(path, allow_pickle=True)
        if recording.shape[0] <= EVENT_ROW:
            raise ValueError(f"{path} does not contain event row {EVENT_ROW}")
        if np.any(recording[EVENT_ROW] == MICROSLEEP_EVENT):
            recordings.append(path)

    if not recordings:
        raise ValueError("No recordings containing Type B microsleeps were found")

    subject_data = []
    subject_labels = []
    current_subject = None
    current_data = []
    current_labels = []

    for path in recordings:
        subject_id = path.name[:5]
        if current_subject is None:
            current_subject = subject_id
        elif subject_id != current_subject:
            subject_data.append(current_data)
            subject_labels.append(current_labels)
            current_subject = subject_id
            current_data = []
            current_labels = []

        recording = np.load(path, allow_pickle=True)
        events = recording[EVENT_ROW]
        for start, end in _segment_bounds(events, WAKE_EVENT):
            if end > start:
                current_data.append(recording[:EEG_CHANNEL_COUNT, start:end])
                current_labels.append(0)
        for start, end in _segment_bounds(events, MICROSLEEP_EVENT):
            if end > start:
                current_data.append(recording[:EEG_CHANNEL_COUNT, start:end])
                current_labels.append(1)

    subject_data.append(current_data)
    subject_labels.append(current_labels)
    return subject_data, subject_labels


def merge_data(data_dir="raw/data"):
    """Return epoch arrays, labels, and the number of epochs per subject."""
    subject_data, subject_labels = load_subjects(data_dir)
    lengths = [len(subject) for subject in subject_data]

    data = []
    labels = []
    for subject, subject_label in zip(subject_data, subject_labels):
        data.extend(subject)
        labels.extend(subject_label)
    return data, labels, lengths


def split_epochs(data, labels, subject_lengths, chunk_size):
    """Split each label-constrained recording segment into fixed-size epochs.

    Trailing samples that do not form a complete epoch are discarded, matching
    the preprocessing used for the reported experiments.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if len(data) != len(labels):
        raise ValueError("data and labels must have the same length")

    source_lengths = [int(length) for length in subject_lengths]
    output_lengths = list(source_lengths)
    epoch_data = []
    epoch_labels = []
    subject_index = 0
    subject_start = 0
    subject_end = source_lengths[0] if source_lengths else 0

    for sample_index, sample in enumerate(data):
        while sample_index >= subject_end:
            subject_index += 1
            if subject_index >= len(source_lengths):
                raise ValueError("subject_lengths does not cover all samples")
            subject_start = subject_end
            subject_end += source_lengths[subject_index]

        sample = np.asarray(sample)
        segment_count = sample.shape[1] // chunk_size
        for segment_index in range(segment_count):
            start = segment_index * chunk_size
            epoch_data.append(sample[:, start:start + chunk_size])
            epoch_labels.append(labels[sample_index])
        output_lengths[subject_index] += segment_count - 1

    if sum(output_lengths) != len(epoch_data):
        raise RuntimeError("Inconsistent subject counts after epoch splitting")
    return epoch_data, epoch_labels, output_lengths
