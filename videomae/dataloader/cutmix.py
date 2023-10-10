
import tensorflow as tf
from tensorflow.keras import layers

class VideoCutMix(layers.Layer):

    def __init__(self, alpha=1.0, seed=None, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.seed = seed

    @staticmethod
    def _sample_from_beta(alpha, beta, shape):
        sample_alpha = tf.random.gamma(shape, 1.0, beta=alpha)
        sample_beta = tf.random.gamma(shape, 1.0, beta=beta)
        return sample_alpha / (sample_alpha + sample_beta)

    def _cutmix_labels(self, labels, lambda_sample, permutation_order):
        cutout_labels = tf.gather(labels, permutation_order)
        lambda_sample = tf.reshape(lambda_sample, [-1, 1])
        labels = lambda_sample * labels + (1.0 - lambda_sample) * cutout_labels
        return labels
    
    def _cutmix_samples(self, videos):
        input_shape = tf.unstack(tf.shape(videos))
        batch_size, num_frame, video_height, video_width = (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        )

        permutation_order = tf.random.shuffle(tf.range(0, batch_size), seed=self.seed)
        lambda_sample = VideoCutMix._sample_from_beta(self.alpha, self.alpha, (batch_size, num_frame))

        ratio = tf.math.sqrt(1 - lambda_sample)
        cut_height = tf.cast(
            ratio * tf.cast(video_height, dtype=tf.float32), dtype=tf.int32
        )
        cut_width = tf.cast(
            ratio * tf.cast(video_height, dtype=tf.float32), dtype=tf.int32
        )

        random_center_height = tf.random.uniform(
            shape=[batch_size, num_frame], minval=0, maxval=video_height, dtype=tf.int32
        )
        random_center_width = tf.random.uniform(
            shape=[batch_size, num_frame], minval=0, maxval=video_width, dtype=tf.int32
        )

        def cutmix_frame(args):
            current_frame, center_x, center_y, width, height, fill_value = args
            return self.fill_rectangle(current_frame, center_x, center_y, width, height, fill_value)

        cutmix_videos = tf.vectorized_map(
            cutmix_frame,
            (
                videos, 
                random_center_width, 
                random_center_height, 
                cut_width, 
                cut_height, 
                tf.gather(videos, permutation_order)
            )
        )

        bounding_box_area = cut_height * cut_width
        lambda_sample = 1.0 - tf.reduce_mean(bounding_box_area / (video_height * video_width), axis=-1)
        lambda_sample = tf.cast(lambda_sample, dtype=tf.float32)

        return cutmix_videos, lambda_sample, permutation_order


    def call(self, batch_inputs, training=None):
        bs_videos = tf.cast(batch_inputs[0], dtype=tf.float32) 
        bs_labels = tf.cast(batch_inputs[1], dtype=tf.float32)

        cutmix_videos, lambda_sample, permutation_order = self._cutmix_samples(
            bs_videos
        )
        cutmix_labels = self._cutmix_labels(bs_labels, lambda_sample, permutation_order)

        return [cutmix_videos, cutmix_labels]

    def fill_rectangle(
        self, videos, centers_x, centers_y, widths, heights, fill_values
    ):
        videos_shape = tf.shape(videos)
        videos_height = videos_shape[1]
        videos_width = videos_shape[2]

        xywh = tf.stack([centers_x, centers_y, widths, heights], axis=1)
        xywh = tf.cast(xywh, tf.float32)
        corners = self.convert_format(xywh)
        
        mask_shape = (videos_width, videos_height)
        is_rectangle = self.corners_to_mask(corners, mask_shape)
        is_rectangle = tf.expand_dims(is_rectangle, -1)
        videos = tf.where(is_rectangle, fill_values, videos)
        return videos

    def convert_format(self, boxes):
        boxes = tf.cast(boxes, dtype=tf.float32)
        x, y, width, height, rest = tf.split(boxes, [1, 1, 1, 1, -1], axis=-1)
        results = tf.concat(
            [
                x - width / 2.0,
                y - height / 2.0,
                x + width / 2.0,
                y + height / 2.0,
                rest,
            ],
            axis=-1,
        )
        return results

    def _axis_mask(self, starts, ends, mask_len):
        # index range of axis
        batch_size = tf.shape(starts)[0]
        axis_indices = tf.range(mask_len, dtype=starts.dtype)
        axis_indices = tf.expand_dims(axis_indices, 0)
        axis_indices = tf.tile(axis_indices, [batch_size, 1])

        # mask of index bounds
        axis_mask = tf.greater_equal(axis_indices, starts) & tf.less(axis_indices, ends)
        return axis_mask

    def corners_to_mask(self, bounding_boxes, mask_shape):
        mask_width, mask_height = mask_shape
        x0, y0, x1, y1 = tf.split(bounding_boxes, [1, 1, 1, 1], axis=-1)
        w_mask = self._axis_mask(x0, x1, mask_width)
        h_mask = self._axis_mask(y0, y1, mask_height)
        w_mask = tf.expand_dims(w_mask, axis=1)
        h_mask = tf.expand_dims(h_mask, axis=2)
        masks = tf.logical_and(w_mask, h_mask)
        return masks
    