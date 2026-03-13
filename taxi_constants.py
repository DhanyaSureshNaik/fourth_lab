SCALE_MINMAX_FEATURE_KEYS = [
    'trip_distance',
    'passenger_count',
]

SCALE_01_FEATURE_KEYS = [
    'pickup_hour',
    'pickup_weekday',
]

SCALE_Z_FEATURE_KEYS = []

VOCAB_FEATURE_KEYS = [
    'payment_type',
]

HASH_STRING_FEATURE_KEYS = [
    'pickup_location_id',
    'dropoff_location_id',
]

FEATURE_DOMAINS = {
    'passenger_count': (1, 6),
    'trip_distance'  : (0, 100),
    'pickup_hour'    : (0, 23),
    'pickup_weekday' : (0, 6),
}

LABEL_KEY = 'trip_duration'

HASH_BUCKETS = 300


def transformed_name(key: str) -> str:
    return key + '_xf'
