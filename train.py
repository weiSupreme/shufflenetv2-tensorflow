import tensorflow as tf
import os
#from model import model_fn, RestoreMovingAverageHook
from modifyClassNum import model_fn, RestoreMovingAverageHook
from input_pipeline import Pipeline
tf.logging.set_verbosity('INFO')


"""
The purpose of this script is to train a network.
Evaluation will happen periodically.

To use it just run:
python train.py

Parameters below is for training 0.5x version.
"""

# 1281144/128 = 10008.9375
# so 1 epoch ~ 10000 steps

GPU_TO_USE = '0'
BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 32
NUM_EPOCHS = 100  # set 166 for 1.0x version
TRAIN_DATASET_SIZE = 4189
NUM_STEPS = NUM_EPOCHS * (TRAIN_DATASET_SIZE // BATCH_SIZE)
PARAMS = {
    'train_dataset_path': 'data/tzbxin0122bsyn_train/',
    'val_dataset_path': 'data/tzbxin0122_val/',
    'weight_decay': 4e-5,
    'initial_learning_rate': 0.00625, #0.0625,  # 0.5/8
    'decay_steps': NUM_STEPS,
    'end_learning_rate': 1e-6,
    'model_dir': 'models/test',
    'num_classes': 2,
    'depth_multiplier': '0.33'  # set '1.0' for 1.0x version
}


def get_input_fn(is_training):

    dataset_path = PARAMS['train_dataset_path'] if is_training else PARAMS['val_dataset_path']
    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    batch_size = BATCH_SIZE if is_training else VALIDATION_BATCH_SIZE
    num_epochs = None if is_training else 1

    def input_fn():
        pipeline = Pipeline(
            filenames, is_training,
            batch_size=batch_size,
            num_epochs=num_epochs
        )
        return pipeline.dataset

    return input_fn


session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.visible_device_list = GPU_TO_USE
session_config.gpu_options.allow_growth = True
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=PARAMS['model_dir'], session_config=session_config,
    save_summary_steps=500, save_checkpoints_secs=300,
    log_step_count_steps=50
)


train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)
estimator = tf.estimator.Estimator(model_fn, params=PARAMS, config=run_config)

train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=NUM_STEPS)
eval_spec = tf.estimator.EvalSpec(
    val_input_fn, steps=None, start_delay_secs=90, throttle_secs=90,
    hooks=[RestoreMovingAverageHook(PARAMS['model_dir'])]
)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
