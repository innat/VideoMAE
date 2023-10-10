import tensorflow as tf
from tensorflow.keras import layers


class VideoMixUp(layers.Layer):
    def __init__(self, alpha=0.2, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.seed = seed

    @staticmethod
    def _sample_from_beta(alpha, beta, shape):
        sample_alpha = tf.random.gamma(shape, 1.0, beta=alpha)
        sample_beta = tf.random.gamma(shape, 1.0, beta=beta)
        return sample_alpha / (sample_alpha + sample_beta)

    def _mixup_samples(self, videos):
        batch_size = tf.shape(videos)[0]
        num_frames = tf.shape(videos)[1]
        
        permutation_order = tf.random.shuffle(tf.range(0, batch_size), seed=self.seed)
        
        lambda_sample = VideoMixUp._sample_from_beta(self.alpha, self.alpha, (batch_size,))
        lambda_sample = tf.reshape(lambda_sample, [-1, 1, 1, 1, 1])

        mixup_videos = tf.gather(videos, permutation_order)
        videos = lambda_sample * videos + (1.0 - lambda_sample) * mixup_videos

        return videos, tf.squeeze(lambda_sample), permutation_order

    def _mixup_labels(self, labels, lambda_sample, permutation_order):
        labels_for_mixup = tf.gather(labels, permutation_order)

        lambda_sample = tf.reshape(lambda_sample, [-1, 1])
        labels = lambda_sample * labels + (1.0 - lambda_sample) * labels_for_mixup

        return labels

    def call(self, batch_inputs):
        bs_videos = tf.cast(batch_inputs[0], dtype=tf.float32)  # ALL Video Samples
        bs_labels = tf.cast(batch_inputs[1], dtype=tf.float32)  # ALL Label Samples
   
        mixup_videos, lambda_sample, permutation_order = self._mixup_samples(bs_videos)
        mixup_labels = self._mixup_labels(bs_labels, lambda_sample, permutation_order)
        
        return [mixup_videos, mixup_labels]

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "seed": self.seed,
            }
        )
        return config