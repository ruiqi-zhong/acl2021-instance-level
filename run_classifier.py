# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
import tokenization
import tensorflow as tf
from utils import load_wrapper, write_wrapper
from utils import lm_bucket, data_bucket

flags = tf.flags

FLAGS = flags.FLAGS
checkpoint_suffixes = ['meta', 'index', 'data-00000-of-00001']
SAVE_PER_EPOCH = 3

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The directory that contains the data, should contain data.json, train.tf_record, predict.tf_record")


flags.DEFINE_integer(
    "initialization_seed", 0,
    "The random seed used for initializing the MLP layer")

flags.DEFINE_integer(
    "dataorder_seed", 0,
    "The random seed used for shuffling the data")


flags.DEFINE_string(
    "pretrain_dir", None,
    "directory that contains all information about a pretrained model.")


flags.DEFINE_string(
    'model_size', None,
    'pretraining model size, mini, small, medium, base, large'
)

flags.DEFINE_integer(
    'pretrain_seed', None,
    'pretraining seed'
)

flags.DEFINE_integer(
    'pretrain_steps', 2000000,
    'number of steps the model has pretrained'
)

flags.DEFINE_bool(
    'original', False,
    'whether to use the original pre-trained model for initialization'
)


## Other parameters

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")


flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")



flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool(
    "do_predict", True,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", None, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 64, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 64, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", None, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 4.0,
                   "Total number of training epochs to perform.")


flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_bool("use_tpu", True, "Whether to use TPU or GPU/CPU.")


DEFAULT_TPU_NAME = os.popen('curl http://metadata.google.internal/computeMetadata/v1/instance/hostname -H Metadata-Flavor:Google').read().split('.')[0]

print('using TPU: ', DEFAULT_TPU_NAME)
tf.flags.DEFINE_string(
    "tpu_name", DEFAULT_TPU_NAME,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")


tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


def rewrite_checkpoint_info(output_dir, model_name):
    tmp_f_name = 'tmp.txt'
    with open(tmp_f_name, 'w') as tmp_out_file:
        tmp_out_file.write('model_checkpoint_path: "%s"\n' % model_name)
        tmp_out_file.write('all_model_checkpoint_paths: "%s"\n' % model_name)
    os.system('gsutil mv %s %s' % (tmp_f_name, os.path.join(output_dir, 'checkpoint')))


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100, seed=FLAGS.dataorder_seed)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02, seed=FLAGS.initialization_seed))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100, seed=FLAGS.dataorder_seed)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


def main(_):
  data_dir = os.path.join(data_bucket, FLAGS.data_dir)
  if FLAGS.train_batch_size is None and FLAGS.learning_rate is None:
    hyperparams = load_wrapper(os.path.join(data_dir, 'size2hyperparam.json'))
    FLAGS.train_batch_size, FLAGS.learning_rate = hyperparams[FLAGS.model_size]['bsize'], hyperparams[FLAGS.model_size]['lr']
    hyper_config = 'besthyper'
  else:
    hyper_config = 'lr%fbsize%d' % (FLAGS.learning_rate, FLAGS.train_batch_size)

  if not FLAGS.original:
    pretrain_dir = '{size}/pretrain_seed{pretrain_seed}step{pretrain_steps}'.format(size=FLAGS.model_size, pretrain_seed=FLAGS.pretrain_seed, pretrain_steps=FLAGS.pretrain_steps)
  else:
    pretrain_dir = '{size}/original'.format(size=FLAGS.model_size)
  output_dir = os.path.join(data_dir, 'results', pretrain_dir,
                            'init%ddataorder%d%s' % (FLAGS.initialization_seed, FLAGS.dataorder_seed, hyper_config)) + '/'
  print('output_dir', output_dir)
  lock_path = os.path.join(output_dir, 'lock.json')
  if tf.gfile.Exists(lock_path):
    print('lock ', lock_path, ' already exists. exitting')
    exit(0)
  else:
    write_wrapper('lock', lock_path)

  init_checkpoint_flag = os.path.join(lm_bucket, pretrain_dir, "bert_model.ckpt")
  bert_config_flag = os.path.join(lm_bucket, pretrain_dir, "bert_config.json")

  tf.logging.set_verbosity(tf.logging.INFO)

  data = load_wrapper(os.path.join(data_dir, 'data.json'))
  print('Successfully loaded data')
  print('task', data['task_name'])
  print('number of train examples', len(data['train']))
  print('number of predict examples', len(data['predict']))

  task_name = data['task_name'].lower()
  print('task', task_name)

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, init_checkpoint_flag)

  bert_config = modeling.BertConfig.from_json_file(bert_config_flag)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(output_dir)

  label_list = data['label_list']

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  train_examples = data['train']
  iteration_per_loop = int(len(train_examples) / SAVE_PER_EPOCH / FLAGS.train_batch_size) + 1
  save_checkpoints_steps = iteration_per_loop
  num_train_steps = save_checkpoints_steps * int(FLAGS.num_train_epochs) * SAVE_PER_EPOCH
  num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
  print('iteration per loop: %d' % iteration_per_loop)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=output_dir,
      save_checkpoints_steps=save_checkpoints_steps,
      keep_checkpoint_max=4,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=iteration_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))


  def get_estimator(init_checkpoint_arg):
    model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=init_checkpoint_arg,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)
    return estimator

  estimator = get_estimator(init_checkpoint_arg=init_checkpoint_flag)

  if FLAGS.do_train:
    train_file = os.path.join(data_dir, "train.tf_record")
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_predict:
    predict_drop_remainder = True if FLAGS.use_tpu else False
    num_actual_predict_examples = len(data['predict'])
    predict_file = os.path.join(data_dir, "predict.tf_record")

    steps_of_interest = [i * save_checkpoints_steps for i in range(num_train_steps // save_checkpoints_steps + 1)]
    steps_of_interest.append(num_train_steps)
    for cur_step in steps_of_interest:
      model_name = "model.ckpt-%d" % cur_step
      init_checkpoint_arg = os.path.join(output_dir, model_name)
      all_files_found = True
      for suffix in checkpoint_suffixes:
        if not tf.gfile.Exists('%s.%s' % (init_checkpoint_arg, suffix)):
          all_files_found = False
      if not all_files_found:
        continue
      rewrite_checkpoint_info(output_dir, model_name)
      predict_input_fn = file_based_input_fn_builder(
          input_file=predict_file,
          seq_length=FLAGS.max_seq_length,
          is_training=False,
          drop_remainder=predict_drop_remainder)

      estimator = get_estimator(init_checkpoint_arg=init_checkpoint_arg)

      result = estimator.predict(input_fn=predict_input_fn)
      output_predict_file = os.path.join(output_dir, "test_results%d.tsv" % cur_step)
      with tf.gfile.GFile(output_predict_file, "w") as writer:
        num_written_lines = 0
        tf.logging.info("***** Predict results *****")
        for (i, prediction) in enumerate(result):
          probabilities = prediction["probabilities"]
          if i >= num_actual_predict_examples:
            break
          output_line = "\t".join(
              str(class_probability)
              for class_probability in probabilities) + "\n"
          writer.write(output_line)
          num_written_lines += 1
      assert num_written_lines == num_actual_predict_examples
      for suffix in checkpoint_suffixes:
        os.system('gsutil rm %s.%s' % (init_checkpoint_arg, suffix))


if __name__ == "__main__":
  tf.app.run()
