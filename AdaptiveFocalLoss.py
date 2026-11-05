"""Focal loss used by CTRSN for class-imbalanced microsleep detection."""

import tensorflow as tf


class DynamicFocalLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        initial_alpha=1.0,
        initial_gamma=1.2,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="dynamic_focal_loss",
    ):
        super().__init__(reduction=reduction, name=name)
        self.alpha_value = float(initial_alpha)
        self.gamma_value = float(initial_gamma)

    def call(self, y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = self.alpha_value * tf.pow(1.0 - y_pred, self.gamma_value)
        return tf.reduce_mean(weight * cross_entropy)

    def get_config(self):
        config = super().get_config()
        config.update(
            initial_alpha=self.alpha_value,
            initial_gamma=self.gamma_value,
        )
        return config
