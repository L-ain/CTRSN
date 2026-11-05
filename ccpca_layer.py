"""Trainable class-channel-wise PCA projection layer."""

import numpy as np
import tensorflow as tf


class CCPCALayer(tf.keras.layers.Layer):
    def __init__(self, u0_init, u1_init, strict_orthogonal=False, **kwargs):
        super().__init__(**kwargs)
        u0_init = np.asarray(u0_init, dtype=np.float32)
        u1_init = np.asarray(u1_init, dtype=np.float32)
        if u0_init.ndim != 3 or u0_init.shape != u1_init.shape:
            raise ValueError("u0_init and u1_init must have equal (C, M, P) shapes")

        self.strict_orthogonal = bool(strict_orthogonal)
        self.u0_init = u0_init
        self.u1_init = u1_init
        if self.strict_orthogonal:
            for matrix in (u0_init, u1_init):
                if matrix.shape[-2] < matrix.shape[-1]:
                    raise ValueError("Strict orthogonality requires M >= P")
                if not np.isfinite(matrix).all():
                    raise ValueError("Projection initialization must be finite")
                if np.any(np.linalg.matrix_rank(matrix) < matrix.shape[-1]):
                    raise ValueError("Each channel must have full column rank")

    def build(self, input_shape):
        self.u0 = self.add_weight(
            name="u0_matrix",
            shape=self.u0_init.shape,
            initializer=tf.keras.initializers.Constant(self.u0_init),
            trainable=True,
        )
        self.u1 = self.add_weight(
            name="u1_matrix",
            shape=self.u1_init.shape,
            initializer=tf.keras.initializers.Constant(self.u1_init),
            trainable=True,
        )
        super().build(input_shape)

    @staticmethod
    def _orthogonal_matrix(raw):
        q, r = tf.linalg.qr(raw, full_matrices=False)
        diagonal = tf.linalg.diag_part(r)
        signs = tf.where(diagonal < 0, -tf.ones_like(diagonal), tf.ones_like(diagonal))
        return q * signs[..., tf.newaxis, :]

    def get_projection_matrices(self):
        if self.strict_orthogonal:
            return self._orthogonal_matrix(self.u0), self._orthogonal_matrix(self.u1)
        return self.u0, self.u1

    def call(self, inputs):
        u0, u1 = self.get_projection_matrices()
        x = tf.transpose(inputs, perm=[0, 2, 1])

        z0 = tf.einsum("bcm,cmp->bcp", x, u0)
        z1 = tf.einsum("bcm,cmp->bcp", x, u1)
        r0 = tf.einsum("bcp,cmp->bcm", z0, u0)
        r1 = tf.einsum("bcp,cmp->bcm", z1, u1)

        e0 = tf.reduce_sum(tf.square(x - r0), axis=[1, 2])
        e1 = tf.reduce_sum(tf.square(x - r1), axis=[1, 2])
        weights = tf.nn.softmax(-tf.stack([e0, e1], axis=1), axis=1)

        z0 = z0 * weights[:, 0, tf.newaxis, tf.newaxis]
        z1 = z1 * weights[:, 1, tf.newaxis, tf.newaxis]
        return tf.concat([z0, z1], axis=-1), e0, e1

    def get_config(self):
        config = super().get_config()
        config.update(
            u0_init=self.u0_init.tolist(),
            u1_init=self.u1_init.tolist(),
            strict_orthogonal=self.strict_orthogonal,
        )
        return config
