import glob
import tensorflow as tf

def read_example(serialized_example):
  feature_description = {
    "signal": tf.io.FixedLenFeature((), tf.string),
    "label": tf.io.FixedLenFeature((), tf.float32),
  }
  example = tf.io.parse_single_example(serialized_example, feature_description)
  signal = tf.io.parse_tensor(example["signal"], out_type = float)
  label = example["label"]

  return signal, label

def get_sleep_stage_data(data_dir, batch_size=1024, shuffle_buffer=10000):
  AUTOTUNE = tf.data.AUTOTUNE

  train_nch_paths = glob.glob(f"{data_dir}/train/*.tfrecords")
  val_nch_paths = glob.glob(f"{data_dir}/val/*.tfrecords")
  test_nch_paths = glob.glob(f"{data_dir}/test/*.tfrecords")

  train_ds = tf.data.TFRecordDataset(train_nch_paths).shuffle(
    shuffle_buffer, reshuffle_each_iteration=True).map(
    read_example, num_parallel_calls=AUTOTUNE).batch(
    batch_size).prefetch(AUTOTUNE)

  val_ds = tf.data.TFRecordDataset(val_nch_paths).map(
    read_example, num_parallel_calls=AUTOTUNE)
  val_ds = val_ds.batch(batch_size).prefetch(AUTOTUNE)

  test_ds = tf.data.TFRecordDataset(test_nch_paths).map(      
    read_example, num_parallel_calls=AUTOTUNE)
  test_ds = test_ds.batch(batch_size).prefetch(AUTOTUNE)

  return train_ds, val_ds, test_ds