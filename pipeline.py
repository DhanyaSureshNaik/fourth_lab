import os
import pprint
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_data_validation as tfdv
import tensorflow_model_analysis as tfma
import ml_metadata as mlmd

from tensorflow_metadata.proto.v0 import schema_pb2
from ml_metadata.proto import metadata_store_pb2
from sklearn.feature_selection import SelectKBest, f_classif
from tfx import v1 as tfx
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext

import taxi_constants as C

pp = pprint.PrettyPrinter()

_PIPELINE_ROOT     = './pipeline_taxi'
_DATA_ROOT         = './data/taxi'
_DATA_RAW          = os.path.join(_DATA_ROOT, 'raw')
_DATA_SELECTED     = os.path.join(_DATA_ROOT, 'selected')
_SERVING_DIR       = './data/taxi/serving'
_SERVING_MODEL_DIR = './serving_model/taxi'
_SCHEMA_DIR        = os.path.join(_PIPELINE_ROOT, 'updated_schema')
_RAW_CSV           = os.path.join(_DATA_RAW,     'taxi_trips.csv')
_SELECTED_CSV      = os.path.join(_DATA_SELECTED, 'taxi_trips.csv')
_SERVING_CSV       = os.path.join(_SERVING_DIR,   'serving_dataset.csv')

for d in [_DATA_RAW, _DATA_SELECTED, _SERVING_DIR,
          _SERVING_MODEL_DIR, _SCHEMA_DIR]:
    os.makedirs(d, exist_ok=True)


def generate_data(path, n=50_000):
    np.random.seed(42)
    dist     = np.random.exponential(3.0, n).clip(0.1, 30).astype(np.float32)
    pax      = np.random.randint(1, 7, n).astype(np.float32)
    hour     = np.random.randint(0, 24, n).astype(np.float32)
    weekday  = np.random.randint(0, 7, n).astype(np.float32)
    pay_type = np.random.choice(['Credit Card', 'Cash', 'No Charge', 'Dispute'], n)
    pickup   = np.random.randint(1, 264, n).astype(str)
    dropoff  = np.random.randint(1, 264, n).astype(str)
    rush     = (np.where((hour >= 7)  & (hour <= 9),  180, 0) +
                np.where((hour >= 16) & (hour <= 19), 240, 0))
    duration = (dist * 200 + rush +
                np.random.normal(0, 60, n)).clip(60, 7200).astype(int)
    df = pd.DataFrame({
        'passenger_count'    : pax,
        'trip_distance'      : dist,
        'pickup_hour'        : hour,
        'pickup_weekday'     : weekday,
        'payment_type'       : pay_type,
        'pickup_location_id' : pickup,
        'dropoff_location_id': dropoff,
        'trip_duration'      : duration,
    })
    df.to_csv(path, index=False)
    print(f'[data] Generated {n:,} rows → {path}')
    return df


if not os.path.exists(_RAW_CSV):
    df_raw = generate_data(_RAW_CSV)
else:
    print(f'[data] Found existing dataset at {_RAW_CSV}')
    df_raw = pd.read_csv(_RAW_CSV)

print(df_raw.head())
print(df_raw['trip_duration'].describe())


NUMERIC_COLS = ['passenger_count', 'trip_distance', 'pickup_hour', 'pickup_weekday']
STRING_COLS  = ['payment_type', 'pickup_location_id', 'dropoff_location_id']
LABEL_COL    = 'trip_duration'
TOP_K        = 4

print('\n[feature-selection] Running SelectKBest ...')
df_num = df_raw[NUMERIC_COLS].copy()
y      = df_raw[LABEL_COL].values

selector      = SelectKBest(f_classif, k=TOP_K)
selector.fit(df_num.values, y)
features_mask = selector.scores_ >= sorted(selector.scores_)[-TOP_K:][0]

selected_numeric = list(df_num.columns[features_mask])
all_selected     = selected_numeric + STRING_COLS + [LABEL_COL]

result_df = pd.DataFrame({'Column': NUMERIC_COLS, 'Retain': features_mask})
print(result_df.to_string(index=False))
print(f'\nSelected features: {all_selected}')

df_selected = df_raw[all_selected]
df_selected.to_csv(_SELECTED_CSV, index=False)
print(f'[feature-selection] Saved reduced dataset → {_SELECTED_CSV}')

df_raw.drop(columns=LABEL_COL).head(100).to_csv(_SERVING_CSV, index=False)
print(f'[feature-selection] Saved serving snapshot → {_SERVING_CSV}')


print('\n[pipeline] Initializing InteractiveContext ...')
context = InteractiveContext(pipeline_root=_PIPELINE_ROOT)


print('\n[pipeline] Running ExampleGen ...')
example_gen = tfx.components.CsvExampleGen(input_base=_DATA_SELECTED)
context.run(example_gen)

artifact = example_gen.outputs['examples'].get()[0]
print(f'  splits : {artifact.split_names}')
print(f'  uri    : {artifact.uri}')


print('\n[pipeline] Running StatisticsGen (pass 1) ...')
statistics_gen = tfx.components.StatisticsGen(
    examples=example_gen.outputs['examples'])
context.run(statistics_gen)


print('\n[pipeline] Running SchemaGen ...')
schema_gen = tfx.components.SchemaGen(
    statistics=statistics_gen.outputs['statistics'])
context.run(schema_gen)

schema_uri = schema_gen.outputs['schema']._artifacts[0].uri
schema     = tfdv.load_schema_text(os.path.join(schema_uri, 'schema.pbtxt'))
print('  Inferred schema loaded.')


print('\n[pipeline] Curating schema ...')

for feat_name, (lo, hi) in C.FEATURE_DOMAINS.items():
    tfdv.set_domain(
        schema, feat_name,
        schema_pb2.IntDomain(name=feat_name, min=lo, max=hi))
    print(f'  Set domain for {feat_name}: [{lo}, {hi}]')

schema.default_environment.append('TRAINING')
schema.default_environment.append('SERVING')
tfdv.get_feature(schema, C.LABEL_KEY).not_in_environment.append('SERVING')
print(f'  Environments: {list(schema.default_environment)}')
print(f'  "{C.LABEL_KEY}" excluded from SERVING environment.')

stats_options     = tfdv.StatsOptions(schema=schema, infer_type_from_schema=True)
serving_stats     = tfdv.generate_statistics_from_csv(
    _SERVING_CSV, stats_options=stats_options)
serving_anomalies = tfdv.validate_statistics(
    serving_stats, schema=schema, environment='SERVING')
serving_ok = not serving_anomalies.anomaly_info
print(f'  Serving anomaly check: {"✓ No anomalies" if serving_ok else "✗ Anomalies detected"}')

schema_file = os.path.join(_SCHEMA_DIR, 'schema.pbtxt')
tfdv.write_schema_text(schema, schema_file)
print(f'  Curated schema saved → {schema_file}')


print('\n[pipeline] Running ImportSchemaGen ...')
user_schema_importer = tfx.components.ImportSchemaGen(schema_file=schema_file)
context.run(user_schema_importer, enable_cache=False)
print(f'  Schema artifact URI: '
      f'{user_schema_importer.outputs["schema"].get()[0].uri}')


print('\n[pipeline] Running StatisticsGen (pass 2 with curated schema) ...')
statistics_gen_updated = tfx.components.StatisticsGen(
    examples=example_gen.outputs['examples'],
    schema=user_schema_importer.outputs['schema'])
context.run(statistics_gen_updated)


print('\n[pipeline] Running ExampleValidator ...')
example_validator = tfx.components.ExampleValidator(
    statistics=statistics_gen_updated.outputs['statistics'],
    schema=user_schema_importer.outputs['schema'])
context.run(example_validator)
print(f'  Anomalies artifact: {example_validator.outputs["anomalies"].get()[0].uri}')


print('\n[pipeline] Running Transform ...')
tf.get_logger().setLevel('ERROR')

transform = tfx.components.Transform(
    examples=example_gen.outputs['examples'],
    schema=user_schema_importer.outputs['schema'],
    module_file=os.path.abspath('taxi_transform.py'))
context.run(transform)

tg_uri = transform.outputs['transform_graph'].get()[0].uri
print(f'  transform_graph contents: {os.listdir(tg_uri)}')


print('\n[pipeline] Running Trainer ...')
trainer = tfx.components.Trainer(
    module_file=os.path.abspath('taxi_trainer.py'),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=user_schema_importer.outputs['schema'],
    train_args=tfx.proto.TrainArgs(num_steps=500),
    eval_args=tfx.proto.EvalArgs(num_steps=100))
context.run(trainer)
print(f'  Model saved at: {trainer.outputs["model"].get()[0].uri}')


print('\n[pipeline] Running Evaluator ...')
eval_config = tfma.EvalConfig(
    model_specs=[
        tfma.ModelSpec(
            label_key=C.transformed_name(C.LABEL_KEY),
            signature_name='serving_default',
            preprocessing_function_names=['tft_layer'],
        )
    ],
    slicing_specs=[
        tfma.SlicingSpec(),
        tfma.SlicingSpec(feature_keys=['payment_type_xf']),
    ],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name='MeanSquaredError'),
                tfma.MetricConfig(class_name='MeanAbsoluteError'),
            ],
            thresholds={
                'mean_absolute_error': tfma.MetricThreshold(
                    value_threshold=tfma.GenericValueThreshold(
                        upper_bound={'value': 0.15}),
                    change_threshold=tfma.GenericChangeThreshold(
                        direction=tfma.MetricDirection.LOWER_IS_BETTER,
                        relative={'value': 0.1}),
                )
            },
        )
    ],
)

evaluator = tfx.components.Evaluator(
    examples=example_gen.outputs['examples'],
    model=trainer.outputs['model'],
    eval_config=eval_config)
context.run(evaluator)

blessing = evaluator.outputs['blessing'].get()[0]
print(f'  Model blessed: {blessing.get_string_custom_property("blessed")}')


print('\n[pipeline] Running Pusher ...')
pusher = tfx.components.Pusher(
    model=trainer.outputs['model'],
    model_blessing=evaluator.outputs['blessing'],
    push_destination=tfx.proto.PushDestination(
        filesystem=tfx.proto.PushDestination.Filesystem(
            base_directory=_SERVING_MODEL_DIR)))
context.run(pusher)
print(f'  Pushed model URI: {pusher.outputs["pushed_model"].get()[0].uri}')
print('\n[pipeline] ✓ Pipeline complete.')


print('\n[mlmd] Querying ML Metadata store ...')

connection_config = context.metadata_connection_config
store    = mlmd.MetadataStore(connection_config)
base_dir = connection_config.sqlite.filename_uri.split('metadata.sqlite')[0]


def display_artifact_table(artifact_list, base_dir, store):
    type_map = {t.id: t.name for t in store.get_artifact_types()}
    print(f'\n  {"ID":>4}  {"Type":<22}  URI')
    print(f'  {"--":>4}  {"----":<22}  ---')
    for a in artifact_list:
        rel = a.uri.replace(base_dir, './')
        print(f'  {a.id:>4}  {type_map.get(a.type_id, "?"):<22}  {rel}')


def get_parent_artifacts(store, artifact):
    artifact_events = store.get_events_by_artifact_ids([artifact.id])
    output_exec_ids = {
        e.execution_id for e in artifact_events
        if e.type == metadata_store_pb2.Event.OUTPUT
    }
    if not output_exec_ids:
        return []
    exec_events = store.get_events_by_execution_ids(list(output_exec_ids))
    parent_ids  = {
        e.artifact_id for e in exec_events
        if e.type == metadata_store_pb2.Event.INPUT
    }
    return store.get_artifacts_by_id(list(parent_ids))


print('\n  All artifact types:')
for t in store.get_artifact_types():
    print(f'    id={t.id:<4} name={t.name}')

print('\n  Schema artifacts:')
display_artifact_table(store.get_artifacts_by_type('Schema'), base_dir, store)

print('\n  TransformGraph artifacts:')
tg_artifacts = store.get_artifacts_by_type('TransformGraph')
display_artifact_table(tg_artifacts, base_dir, store)

if tg_artifacts:
    tg      = tg_artifacts[0]
    parents = get_parent_artifacts(store, tg)
    print(f'\n  Parent artifacts of TransformGraph (id={tg.id}):')
    display_artifact_table(parents, base_dir, store)

stats_artifacts = store.get_artifacts_by_type('ExampleStatistics')
if stats_artifacts:
    last_stats = stats_artifacts[-1]
    parents    = get_parent_artifacts(store, last_stats)
    print(f'\n  Parent artifacts of ExampleStatistics (id={last_stats.id}):')
    display_artifact_table(parents, base_dir, store)

print('\n[mlmd] ✓ Lineage tracking complete.')
