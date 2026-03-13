import tensorflow as tf
import tensorflow_transform as tft
import taxi_constants as C


def preprocessing_fn(inputs):
    outputs = {}

    for key in C.SCALE_MINMAX_FEATURE_KEYS:
        outputs[C.transformed_name(key)] = tft.scale_by_min_max(inputs[key])

    for key in C.SCALE_01_FEATURE_KEYS:
        outputs[C.transformed_name(key)] = tft.scale_to_0_1(inputs[key])

    for key in C.SCALE_Z_FEATURE_KEYS:
        outputs[C.transformed_name(key)] = tft.scale_to_z_score(
            tf.cast(inputs[key], tf.float32))

    for key in C.VOCAB_FEATURE_KEYS:
        outputs[C.transformed_name(key)] = tft.compute_and_apply_vocabulary(
            inputs[key],
            num_oov_buckets=1,
        )

    for key in C.HASH_STRING_FEATURE_KEYS:
        outputs[C.transformed_name(key)] = tft.hash_strings(
            inputs[key], hash_buckets=C.HASH_BUCKETS)

    outputs[C.transformed_name(C.LABEL_KEY)] = tft.scale_to_0_1(
        tf.cast(inputs[C.LABEL_KEY], tf.float32))

    return outputs
