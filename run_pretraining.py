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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import modeling
import optimization
from distutils.util import strtobool
import argparse
import chainer
#chainer.set_debug(True)
from chainer import training
from chainer import functions as F
from chainer.datasets import PickleDataset
from chainer.training import extensions
from chainer import serializers
import numpy as np


def get_arguments():
    parser = argparse.ArgumentParser(description='Arxiv')
    # Required parameters
    parser.add_argument(
        "--bert_config_file", type=str, default=None)
    parser.add_argument(
        "--input_file", type=str, default=None)

    parser.add_argument(
        "--output_dir", type=str, default=None)

    # Other parameters
    parser.add_argument(
        "--init_checkpoint", type=str, default=None)

    parser.add_argument("--max_seq_length", type=int, default=128)

    parser.add_argument("--max_predictions_per_seq", type=int, default=20)

    parser.add_argument("--do_train", type=strtobool, default=False)

    parser.add_argument("--do_eval", type=strtobool, default=False)

    parser.add_argument("--train_batch_size", type=int, default=32)

    parser.add_argument("--eval_batch_size", type=int, default=8)

    parser.add_argument("--learning_rate", type=float, default=5e-5)

    parser.add_argument("--num_train_steps", type=int, default=100000)

    parser.add_argument("--num_warmup_steps", type=int, default=10000)

    parser.add_argument("--save_checkpoints_steps", type=int, default=1000)

    parser.add_argument("--iterations_per_loop", type=int, default=1000)

    parser.add_argument("--max_eval_steps", type=int, default=100)

    parser.add_argument("--gpu", default=0, type=int)

    parser.add_argument("--use_tpu", default=False, type=strtobool)

    args = parser.parse_args()
    return args


FLAGS = get_arguments()


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" %
                            (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]
        next_sentence_labels = features["next_sentence_labels"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
             bert_config, model.get_sequence_output(), model.get_embedding_table(),
             masked_lm_positions, masked_lm_ids, masked_lm_weights)

        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs) = get_next_sentence_output(
             bert_config, model.get_pooled_output(), next_sentence_labels)

        total_loss = masked_lm_loss + next_sentence_loss

        tvars = tf.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map,
             initialized_variable_names) = modeling.get_assigment_map_from_checkpoint(
                 tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(
                        init_checkpoint, assignment_map)
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

            def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                          masked_lm_weights, next_sentence_example_loss,
                          next_sentence_log_probs, next_sentence_labels):
                """Computes the loss and accuracy of the model."""
                masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                                 [-1, masked_lm_log_probs.shape[-1]])
                masked_lm_predictions = tf.argmax(
                    masked_lm_log_probs, axis=-1, output_type=tf.int32)
                masked_lm_example_loss = tf.reshape(
                    masked_lm_example_loss, [-1])
                masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
                masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
                masked_lm_accuracy = tf.metrics.accuracy(
                    labels=masked_lm_ids,
                    predictions=masked_lm_predictions,
                    weights=masked_lm_weights)
                masked_lm_mean_loss = tf.metrics.mean(
                    values=masked_lm_example_loss, weights=masked_lm_weights)

                next_sentence_log_probs = tf.reshape(
                    next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
                next_sentence_predictions = tf.argmax(
                    next_sentence_log_probs, axis=-1, output_type=tf.int32)
                next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
                next_sentence_accuracy = tf.metrics.accuracy(
                    labels=next_sentence_labels, predictions=next_sentence_predictions)
                next_sentence_mean_loss = tf.metrics.mean(
                    values=next_sentence_example_loss)

                return {
                    "masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss,
                    "next_sentence_accuracy": next_sentence_accuracy,
                    "next_sentence_loss": next_sentence_mean_loss,
                }

            eval_metrics = (metric_fn, [
                masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                masked_lm_weights, next_sentence_example_loss,
                next_sentence_log_probs, next_sentence_labels
            ])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            raise ValueError(
                "Only TRAIN and EVAL modes are supported: %s" % (mode))

        return output_spec

    return model_fn


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = - \
            tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
    """Get loss and log probs for the next sentence prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.variable_scope("cls/seq_relationship"):
        output_weights = tf.get_variable(
            "output_weights",
            shape=[2, bert_config.hidden_size],
            initializer=modeling.create_initializer(bert_config.initializer_range))
        output_bias = tf.get_variable(
            "output_bias", shape=[2], initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        name_to_features = {
            "input_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":
                tf.FixedLenFeature([max_seq_length], tf.int64),
            "masked_lm_positions":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids":
                tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_weights":
                tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
            "next_sentence_labels":
                tf.FixedLenFeature([1], tf.int64),
        }

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(input_files))

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))

            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                tf.contrib.data.parallel_interleave(
                    tf.data.TFRecordDataset,
                    sloppy=is_training,
                    cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                num_parallel_batches=num_cpu_threads,
                drop_remainder=True))
        return d

    return input_fn


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


def _load_data_using_dataset_api(input_file):
    print(input_file)
    return PickleDataset(open(input_file, 'rb'))

def fuga(value):
    return value+2

def main():
    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if not os.path.isdir(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    def _get_text_file(text_dir):
        import glob
        #file_list = glob.glob(f'{text_dir}/**/*')
        # seqが512
        #file_list = ['/nfs/ai16storage01/sec/akp2/1706nasubi/inatomi/benchmark/bert-chainer/data/wiki_data_pickle/all']
        # seqが128
        file_list = ['/nfs/ai16storage01/sec/akp2/1706nasubi/inatomi/benchmark/bert-chainer/data/wiki_data_pickle/all_seq128']
        # debug
        #file_list = ['/nfs/ai16storage01/sec/akp2/1706nasubi/inatomi/benchmark/bert-chainer/data/wiki_data_pickle/AA/wiki_00']
        files = ",".join(file_list)
        return files
    input_files = _get_text_file(FLAGS.input_file).split(',')

   #  model_fn = model_fn_builder(
   #      bert_config=bert_config,
   #      init_checkpoint=FLAGS.init_checkpoint,
   #      learning_rate=FLAGS.learning_rate,
   #      num_train_steps=FLAGS.num_train_steps,
   #      num_warmup_steps=FLAGS.num_warmup_steps,
   #      use_tpu=FLAGS.use_tpu,
   #      use_one_hot_embeddings=FLAGS.use_tpu)

    if FLAGS.do_train:
        input_files = input_files
    bert = modeling.BertModel(config=bert_config)
    model = modeling.BertPretrainer(bert)
    if FLAGS.init_checkpoint:
        serializers.load_npz(FLAGS.init_checkpoint, model)
        model = modeling.BertPretrainer(model.bert)
    if FLAGS.gpu >= 0:
        pass
        #chainer.backends.cuda.get_device_from_id(FLAGS.gpu).use()
        #model.to_gpu()

    if FLAGS.do_train:
        """chainerでのpretrainを記述。BERTClassificationに変わるものを作成し、BERTの出力をこねこねしてmodel_fnが返すものと同じものを返すようにすれば良いか?"""
        # Adam with weight decay only for 2D matrices
        optimizer = optimization.WeightDecayForMatrixAdam(
            alpha=1.,  # ignore alpha. instead, use eta as actual lr
            eps=1e-6, weight_decay_rate=0.01)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.GradientClipping(1.))

        """ ConcatenatedDatasetはon memolyなため、巨大データセットのPickleを扱えない
        input_files = sorted(input_files)[:len(input_files) // 2]
        input_files = sorted(input_files)[:200]
        import concurrent.futures
        train_examples = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for train_exapmle in executor.map(_load_data_using_dataset_api, input_files):
                train_examples.append(train_exapmle)
        train_examples = ConcatenatedDataset(*train_examples)
        """
        train_examples = _load_data_using_dataset_api(input_files[0])

        train_iter = chainer.iterators.SerialIterator(
            train_examples, FLAGS.train_batch_size)
        converter = Converter()
        if False:
            updater = training.updaters.StandardUpdater(
                train_iter, optimizer,
                converter=converter,
                device=FLAGS.gpu)
        else:
            updater = training.updaters.ParallelUpdater(
                iterator=train_iter,
                optimizer=optimizer,
                converter=converter,
                # The device of the name 'main' is used as a "master", while others are
                # used as slaves. Names other than 'main' are arbitrary.
                devices={'main': 0,
                         '1': 1,
                         '2': 2,
                         '3': 3,
                         '4': 4,
                         '5': 5,
                         '6': 6,
                         '7': 7,
                         },
            )
        # learning rate (eta) scheduling in Adam
        num_warmup_steps = FLAGS.num_warmup_steps
        num_train_steps = FLAGS.num_train_steps
        trainer = training.Trainer(
            updater, (num_train_steps, 'iteration'), out=FLAGS.output_dir)
        lr_decay_init = FLAGS.learning_rate * \
            (num_train_steps - num_warmup_steps) / num_train_steps
        trainer.extend(extensions.LinearShift(  # decay
            'eta', (lr_decay_init, 0.), (num_warmup_steps, num_train_steps)))
        trainer.extend(extensions.WarmupShift(  # warmup
            'eta', 0., num_warmup_steps, FLAGS.learning_rate))
        trainer.extend(extensions.observe_value(
            'eta', lambda trainer: trainer.updater.get_optimizer('main').eta),
            trigger=(50, 'iteration'))  # logging

        trainer.extend(extensions.snapshot_object(
            model, 'seq_128_model_snapshot_iter_{.updater.iteration}.npz'),
            trigger=(1000, 'iteration'))
        trainer.extend(extensions.LogReport(
            trigger=(1, 'iteration')))
        #trainer.extend(extensions.PlotReport(
        #    [
        #        'main/next_sentence_loss',
        #        'main/next_sentence_accuracy',
        #     ], (3, 'iteration'), file_name='next_sentence.png'))
        #trainer.extend(extensions.PlotReport(
        #    [
        #        'main/masked_lm_loss',
        #        'main/masked_lm_accuracy',
        #     ], (3, 'iteration'), file_name='masked_lm.png'))
        trainer.extend(extensions.PlotReport(
            y_keys=[
                'main/loss',
                'main/next_sentence_loss',
                'main/next_sentence_accuracy',
                'main/masked_lm_loss',
                'main/masked_lm_accuracy',
             ], x_key='iteration', trigger=(100, 'iteration'), file_name='loss.png'))
        trainer.extend(extensions.PrintReport(
            ['iteration',
             'main/loss',
             'main/masked_lm_loss', 'main/masked_lm_accuracy',
             'main/next_sentence_loss', 'main/next_sentence_accuracy',
             'elapsed_time']))
        trainer.extend(extensions.ProgressBar(update_interval=20))

        trainer.run()

    if FLAGS.do_eval:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_input_fn = input_fn_builder(
            input_files=input_files,
            max_seq_length=FLAGS.max_seq_length,
            max_predictions_per_seq=FLAGS.max_predictions_per_seq,
            is_training=False)

        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


class Converter(object):
    """Converts examples to features, and then batches and to_gpu."""

    def __init__(self):
        pass

    def __call__(self, examples, gpu):
        return self.convert_examples_to_features(examples, gpu)

    def convert_examples_to_features(self, examples, gpu):
        """Loads a data file into a list of `InputBatch`s.

        Args:
          examples: A list of examples (`InputExample`s).
          gpu: int. The gpu device id to be used. If -1, cpu is used.

        """
        return self.make_batch(examples, gpu)

    def make_batch(self, features, gpu):
        """Creates a concatenated batch from a list of data and to_gpu."""

        all_input_ids = []
        all_input_mask = []
        all_segment_ids = []
        all_masked_lm_positions = []
        all_masked_lm_ids = []
        all_masked_lm_weights = []
        all_next_sentence_labels = []

        for feature in features:
            all_input_ids.append(np.array(feature['input_ids'], 'i'))
            all_input_mask.append(np.array(feature['input_mask'], 'i'))
            all_segment_ids.append(np.array(feature['segment_ids'], 'i'))
            all_masked_lm_positions.append(np.array(feature['masked_lm_positions'], 'i'))
            all_masked_lm_ids.append(np.array(feature['masked_lm_ids'], 'i'))
            all_masked_lm_weights.append(np.array(feature['masked_lm_weights'], 'f'))
            all_next_sentence_labels.append(np.array([feature['next_sentence_labels']], 'i'))

        def stack_and_to_gpu(data_list, astype):
            sdata = F.pad_sequence(
                data_list, length=None, padding=0).array.astype(astype)
            return chainer.dataset.to_device(gpu, sdata)

        batch_input_ids = stack_and_to_gpu(all_input_ids, 'i')
        batch_input_mask = stack_and_to_gpu(all_input_mask, 'i')
        batch_input_segment_ids = stack_and_to_gpu(all_segment_ids, 'i')
        batch_input_masked_lm_positions = stack_and_to_gpu(all_masked_lm_positions, 'i')
        batch_input_masked_lm_ids = stack_and_to_gpu(all_masked_lm_ids, 'i')
        batch_input_masked_lm_weights = stack_and_to_gpu(all_masked_lm_weights, 'f')
        batch_input_next_sentence_labels = stack_and_to_gpu(all_next_sentence_labels, 'i')[:, 0]
        return (batch_input_ids, batch_input_mask,
                batch_input_segment_ids, batch_input_masked_lm_positions, batch_input_masked_lm_ids,
                batch_input_masked_lm_weights, batch_input_next_sentence_labels)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = np.array(input_ids, 'i')
        self.input_mask = np.array(input_mask, 'i')
        self.segment_ids = np.array(segment_ids, 'i')
        self.label_id = np.array([label_id], 'i')  # shape changed


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

if __name__ == "__main__":
    main()
