"""Train and evaluate CTRSN with leave-one-subject-out cross-validation."""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef

import dataloader
from AdaptiveFocalLoss import DynamicFocalLoss
from PCA import initialize_cc_pca
from ccpca_layer import CCPCALayer


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def configure_gpus():
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass


def residual_shrinkage_block(x, filters, kernel_size, dilation_rate, name):
    r = tf.keras.layers.Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        name=f"conv_{name}_1",
    )(x)
    r = tf.keras.layers.BatchNormalization(name=f"bn_{name}_1")(r)
    r = tf.keras.layers.Activation("relu", name=f"relu_{name}_1")(r)

    r = tf.keras.layers.Conv1D(
        filters,
        kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
        name=f"conv_{name}_2",
    )(r)
    r = tf.keras.layers.BatchNormalization(name=f"bn_{name}_2")(r)
    r = tf.keras.layers.Activation("relu", name=f"relu_{name}_2")(r)

    residual_abs = tf.abs(r)
    abs_mean = tf.keras.layers.GlobalAveragePooling1D(name=f"gap_{name}")(residual_abs)
    scales = tf.keras.layers.Dense(
        filters,
        activation=None,
        kernel_initializer="he_normal",
        kernel_regularizer=tf.keras.regularizers.l1(0.001),
        name=f"threshold_dense_{name}_1",
    )(abs_mean)
    scales = tf.keras.layers.BatchNormalization(name=f"threshold_bn_{name}")(scales)
    scales = tf.keras.layers.Activation(
        "relu", name=f"threshold_relu_{name}"
    )(scales)
    scales = tf.keras.layers.Dense(
        filters,
        activation="sigmoid",
        kernel_regularizer=tf.keras.regularizers.l2(0.001),
        name=f"threshold_dense_{name}_2",
    )(scales)
    threshold = tf.keras.layers.Multiply(name=f"threshold_{name}")([abs_mean, scales])

    shrunk = tf.keras.layers.Subtract(name=f"shrink_sub_{name}")([residual_abs, threshold])
    zeros = tf.zeros_like(shrunk)
    shrunk = tf.keras.layers.Maximum(name=f"shrink_max_{name}")([shrunk, zeros])
    residual = tf.keras.layers.Multiply(name=f"sign_restore_{name}")([tf.sign(r), shrunk])

    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = tf.keras.layers.Conv1D(
            filters, 1, padding="same", name=f"shortcut_conv_{name}"
        )(x)
    output = tf.keras.layers.Add(name=f"residual_add_{name}")([residual, shortcut])
    return tf.keras.layers.Activation("relu", name=f"residual_relu_{name}")(output)


def build_ctrsn(input_shape, u0_init, u1_init, strict_orthogonal=False):
    inputs = tf.keras.Input(shape=input_shape, name="eeg_input")
    features, e0, e1 = CCPCALayer(
        u0_init,
        u1_init,
        strict_orthogonal=strict_orthogonal,
        name="cc_pca",
    )(inputs)

    x = tf.keras.layers.Masking(mask_value=0.0, name="masking")(features)
    x = residual_shrinkage_block(x, 16, 3, 1, "block_1")
    x = residual_shrinkage_block(x, 16, 3, 2, "block_2")
    x = residual_shrinkage_block(x, 8, 3, 4, "block_3")
    x = tf.keras.layers.Flatten(name="flatten")(x)
    classification = tf.keras.layers.Dense(
        2, activation="softmax", name="cls_output"
    )(x)
    e0_output = tf.keras.layers.Layer(name="e0_output")(e0)
    e1_output = tf.keras.layers.Layer(name="e1_output")(e1)
    return tf.keras.Model(inputs, [classification, e0_output, e1_output], name="CTRSN")


def conditional_reconstruction_loss_0(y_true, reconstruction_error):
    is_class_0 = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), 0), tf.float32)
    return tf.reduce_mean(is_class_0 * reconstruction_error)


def conditional_reconstruction_loss_1(y_true, reconstruction_error):
    is_class_1 = tf.cast(tf.equal(tf.argmax(y_true, axis=-1), 1), tf.float32)
    return tf.reduce_mean(is_class_1 * reconstruction_error)


def mcc_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.cast(tf.round(y_pred), tf.int32)
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
    tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.float32))
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))
    numerator = tp * tn - fp * fn
    denominator = tf.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return tf.where(tf.equal(denominator, 0.0), 0.0, numerator / denominator)


def prepare_data(data_dir, epoch_samples):
    data, labels, subject_lengths = dataloader.merge_data(data_dir)
    data, labels, subject_lengths = dataloader.split_epochs(
        data, labels, subject_lengths, epoch_samples
    )
    data = np.asarray(data, dtype=np.float32).transpose(0, 2, 1)
    labels = np.asarray(labels, dtype=np.int64)
    if data.ndim != 3:
        raise ValueError("Prepared data must have shape (samples, time, channels)")
    if sum(subject_lengths) != len(labels):
        raise RuntimeError("Subject lengths do not match the number of epochs")
    return data, labels, subject_lengths


def save_results(result_dir, result_name, metrics, subjects):
    if not result_name:
        return
    for metric_name, values in metrics.items():
        if not values or not np.isfinite(values).all():
            raise ValueError(f"Invalid {metric_name} results")
        path = result_dir / metric_name / "0.5-s" / result_name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(",".join(str(value) for value in values))
    subject_path = result_dir / "subjects" / "0.5-s" / result_name
    subject_path.parent.mkdir(parents=True, exist_ok=True)
    with subject_path.open("x", encoding="utf-8") as stream:
        stream.write(",".join(str(subject) for subject in subjects))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="raw/data")
    parser.add_argument("--epoch-samples", type=int, default=100)
    parser.add_argument("--pca-components", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.5e-4)
    parser.add_argument("--lambda-rec", type=float, default=1e-8)
    parser.add_argument("--strict-orthogonal", action="store_true")
    parser.add_argument("--result-name")
    parser.add_argument("--result-dir", default="results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    args = parser.parse_args()
    if args.result_name and Path(args.result_name).name != args.result_name:
        parser.error("--result-name must be a filename")
    return args


def main():
    args = parse_args()
    configure_gpus()
    set_random_seed(args.seed)

    data, labels, subject_lengths = prepare_data(args.data_dir, args.epoch_samples)
    result_dir = Path(args.result_dir)
    metric_values = {
        "phi": [],
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
    }
    subject_results = []
    inference_times = []
    subject_start = 0

    for subject_index, subject_length in enumerate(subject_lengths):
        subject_end = subject_start + int(subject_length)
        test_indices = np.arange(subject_start, subject_end)
        train_indices = np.concatenate(
            (np.arange(0, subject_start), np.arange(subject_end, len(labels)))
        )
        subject_start = subject_end

        if not np.any(labels[test_indices] == 1):
            print(f"Skipping subject {subject_index + 1}: no microsleep epochs")
            continue

        x_train = data[train_indices]
        y_train = labels[train_indices]
        x_test = data[test_indices]
        y_test = labels[test_indices]
        y_train_categorical = tf.keras.utils.to_categorical(y_train, 2)

        projections = initialize_cc_pca(
            x_train,
            y_train,
            n_components=args.pca_components,
            random_state=args.seed,
        )
        model = build_ctrsn(
            data.shape[1:],
            projections[0],
            projections[1],
            strict_orthogonal=args.strict_orthogonal,
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
            loss={
                "cls_output": DynamicFocalLoss(),
                "e0_output": conditional_reconstruction_loss_0,
                "e1_output": conditional_reconstruction_loss_1,
            },
            loss_weights={
                "cls_output": 1.0,
                "e0_output": args.lambda_rec,
                "e1_output": args.lambda_rec,
            },
            metrics={"cls_output": [mcc_metric, "accuracy"]},
        )

        train_targets = {
            "cls_output": y_train_categorical,
            "e0_output": y_train_categorical,
            "e1_output": y_train_categorical,
        }
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="loss",
                patience=10,
                restore_best_weights=True,
            )
        ]
        model.fit(
            x_train,
            train_targets,
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=args.verbose,
        )

        inference_start = time.perf_counter()
        probabilities = model.predict(x_test, verbose=0)[0]
        inference_times.append(time.perf_counter() - inference_start)
        predictions = np.argmax(probabilities, axis=1)

        tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
        values = {
            "phi": matthews_corrcoef(y_test, predictions),
            "accuracy": accuracy_score(y_test, predictions),
            "sensitivity": tp / (tp + fn) if tp + fn else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
        }
        for metric_name, value in values.items():
            metric_values[metric_name].append(float(value))
        subject_results.append(subject_index + 1)
        print(
            f"Subject {subject_index + 1}: "
            f"phi={values['phi']:.4f}, accuracy={values['accuracy']:.4f}, "
            f"sensitivity={values['sensitivity']:.4f}, "
            f"specificity={values['specificity']:.4f}"
        )

    if not subject_results:
        raise RuntimeError("No subjects with microsleep epochs were evaluated")

    for metric_name, values in metric_values.items():
        print(f"Mean {metric_name}: {np.mean(values):.4f}")
    print(f"Mean inference time: {np.mean(inference_times):.4f} s")
    save_results(result_dir, args.result_name, metric_values, subject_results)


if __name__ == "__main__":
    main()
