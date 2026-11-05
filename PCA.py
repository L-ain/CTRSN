"""Class-channel-wise PCA initialization used by the CTRSN projection layer."""

import numpy as np
from sklearn.decomposition import PCA


def initialize_cc_pca(data, labels, n_components=20, random_state=42):
    """Fit one PCA basis per class and EEG channel.

    Args:
        data: Array with shape ``(samples, time, channels)``.
        labels: Binary labels with shape ``(samples,)``.
        n_components: Number of temporal principal components per channel.
        random_state: Seed passed to scikit-learn PCA.

    Returns:
        A dictionary mapping classes 0 and 1 to arrays with shape
        ``(channels, time, n_components)``.
    """
    data = np.asarray(data)
    labels = np.asarray(labels)
    if data.ndim != 3:
        raise ValueError("data must have shape (samples, time, channels)")
    if labels.shape != (data.shape[0],):
        raise ValueError("labels must contain one value per sample")
    if n_components <= 0:
        raise ValueError("n_components must be positive")

    projections = {}
    for class_index in (0, 1):
        class_data = data[labels == class_index]
        if class_data.shape[0] == 0:
            raise ValueError(f"No samples found for class {class_index}")

        while class_data.shape[0] < n_components:
            class_data = np.concatenate((class_data, class_data), axis=0)

        class_components = []
        for channel_index in range(data.shape[2]):
            pca = PCA(n_components=n_components, random_state=random_state)
            pca.fit(class_data[:, :, channel_index])
            class_components.append(pca.components_.T)
        projections[class_index] = np.stack(class_components, axis=0)
    return projections
