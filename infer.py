import os
import tensorflow as tf
import taxi_constants as C

_SERVING_MODEL_DIR = './serving_model/taxi'


def make_example(passenger_count, trip_distance, pickup_hour,
                 pickup_weekday, payment_type, pickup_loc, dropoff_loc):
    return tf.train.Example(features=tf.train.Features(feature={
        'passenger_count'    : tf.train.Feature(
            float_list=tf.train.FloatList(value=[passenger_count])),
        'trip_distance'      : tf.train.Feature(
            float_list=tf.train.FloatList(value=[trip_distance])),
        'pickup_hour'        : tf.train.Feature(
            float_list=tf.train.FloatList(value=[pickup_hour])),
        'pickup_weekday'     : tf.train.Feature(
            float_list=tf.train.FloatList(value=[pickup_weekday])),
        'payment_type'       : tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[payment_type.encode()])),
        'pickup_location_id' : tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[str(pickup_loc).encode()])),
        'dropoff_location_id': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[str(dropoff_loc).encode()])),
    })).SerializeToString()


def load_model(serving_dir):
    versions = sorted(os.listdir(serving_dir))
    if not versions:
        raise FileNotFoundError(
            f'No pushed model found in {serving_dir}. Run pipeline.py first.')
    latest = os.path.join(serving_dir, versions[-1])
    print(f'[infer] Loading model from: {latest}')
    model = tf.saved_model.load(latest)
    return model.signatures['serving_default']


def predict(infer_fn, examples):
    output = infer_fn(examples=tf.constant(examples))
    key    = list(output.keys())[0]
    return output[key].numpy().flatten()


if __name__ == '__main__':
    infer_fn = load_model(_SERVING_MODEL_DIR)

    test_cases = [
        (1, 1.0,  9.0, 1.0, 'Credit Card', 161, 234, 'Short Tue morning'),
        (2, 3.0, 17.0, 4.0, 'Credit Card', 161, 234, 'Friday rush hour'),
        (4, 8.5, 22.0, 5.0, 'Cash',        132,  45, 'Long Sat night'),
    ]

    serialised = [
        make_example(p, d, h, wd, pay, pl, dl)
        for p, d, h, wd, pay, pl, dl, _ in test_cases
    ]
    preds = predict(infer_fn, serialised)

    print(f'\n[infer] Predictions (label scaled to [0, 1]):')
    print(f'  {"Description":<25} {"Scaled prediction":>20}')
    print('  ' + '-' * 47)
    for (_, _, _, _, _, _, _, desc), pred in zip(test_cases, preds):
        print(f'  {desc:<25} {pred:>20.4f}')

    print('\n  Tip: multiply by max trip_duration in training data to get seconds.')
