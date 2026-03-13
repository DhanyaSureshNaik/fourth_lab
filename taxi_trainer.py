import tensorflow as tf
import tensorflow_transform as tft
from tfx import v1 as tfx
from tfx_bsl.public import tfxio

import taxi_constants as C

_EPOCHS     = 5
_BATCH_SIZE = 64


def _input_fn(file_pattern, data_accessor, tf_transform_output, batch_size):
    return data_accessor.tf_dataset_factory(
        file_pattern,
        tfxio.TensorFlowDatasetOptions(
            batch_size=batch_size,
            label_key=C.transformed_name(C.LABEL_KEY)),
        tf_transform_output.transformed_metadata.schema,
    ).repeat()


def _build_keras_model(tf_transform_output):
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    feature_spec.pop(C.transformed_name(C.LABEL_KEY))

    inputs = {}
    for key, spec in feature_spec.items():
        if isinstance(spec, tf.io.FixedLenFeature):
            inputs[key] = tf.keras.Input(
                shape=spec.shape, name=key, dtype=spec.dtype)
        else:
            inputs[key] = tf.keras.Input(
                shape=(1,), name=key, dtype=tf.float32)

    x = tf.keras.layers.concatenate(
        [tf.cast(v, tf.float32) for v in inputs.values()])

    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    output = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae')])
    model.summary()
    return model


def _get_serve_fn(model, tf_transform_output):
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_fn(serialized_tf_examples):
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(C.LABEL_KEY)
        parsed = tf.io.parse_example(serialized_tf_examples, feature_spec)
        transformed = model.tft_layer(parsed)
        return model(transformed)

    return serve_fn


def run_fn(fn_args: tfx.components.FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = _input_fn(
        fn_args.train_files, fn_args.data_accessor,
        tf_transform_output, _BATCH_SIZE)
    eval_dataset = _input_fn(
        fn_args.eval_files, fn_args.data_accessor,
        tf_transform_output, _BATCH_SIZE)

    model = _build_keras_model(tf_transform_output)

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        epochs=_EPOCHS,
        callbacks=[tf.keras.callbacks.TensorBoard(
            log_dir=fn_args.model_run_dir, update_freq='batch')],
    )

    signatures = {
        'serving_default': _get_serve_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
    }
    model.save(fn_args.serving_model_dir,
               save_format='tf', signatures=signatures)
