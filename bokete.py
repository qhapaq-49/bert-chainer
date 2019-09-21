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

import os
import modeling
from modeling import BertConfig, BertModel, BertPretrainer, LayerNormalization3D
import optimization
from distutils.util import strtobool
import chainer
#chainer.set_debug(True)
from chainer import training
from chainer import functions as F
from chainer.datasets import PickleDataset
from chainer.training import extensions
from chainer import serializers
import numpy
from modeling import BertExtracter, BertModel
from tokenization_sentencepiece import FullTokenizer
import sys
from chainer import functions as F
from chainer import links as L
from chainer import initializers
from chainer import cuda
from chainer.datasets import PickleDataset
#from run_pretraining import Converter
import cupy as np

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

def tokenize_with_mask(text, tokenizer, xp, mask_delim="[MASK]", mask_id=6, cls_id=4):
    v_tokens = tokenizer.tokenize(text)
    tokens = tokenizer.convert_tokens_to_ids(v_tokens)
    return xp.reshape(xp.array(tokens).astype(xp.int32), (1, -1)), v_tokens
    

class BertInterpolater(chainer.Chain):
    """
    [MASK]入りのテキストを投げると、MASKを補完したテキストを返してくれる子
    bert ... modeling.BertPretrainer を参照
    tokenizer ... 適当なtokenizer textを突っ込んでidsを返してくれるはず
    """
    def __init__(self, model, tokenizer):
        super(BertInterpolater, self).__init__()
        self.tokenizer = tokenizer
        with self.init_scope():
            self.model = model

    def push_text(self, text):
        tokens, v_tokens = tokenize_with_mask(text, self.tokenizer, self.xp)
        mask_token = self.tokenizer.vocab["[MASK]"]
        self.model.bert.get_sequence_output(tokens, None)
        embedding_table = self.model.bert.get_embedding_table()
        sequence_output = self.model.bert.get_sequence_output(
            tokens,
            None,
            None)
        
        batch_size, seq_length, width = sequence_output.shape        

        flat_sequence_output = self.xp.reshape(sequence_output.data,[batch_size * seq_length, width])

        normed = self.model.layer_norm(self.model.activate(self.model.masked_lm_dense(flat_sequence_output)))
        masked_lm_logits = F.matmul(normed, embedding_table.T) + self.model.mask_bias
        masked_lm_ids = F.argmax(masked_lm_logits, axis=1)
        masked_lm_ids = F.reshape(masked_lm_ids, (batch_size, seq_length, 1))
        trans = masked_lm_ids.data[0]
        trans = cuda.to_cpu(trans)
        # output
        outstr = ""
        print("original = ", text)
        tokens = cuda.to_cpu(tokens[0])
        for i in range(len(trans)):
            if tokens[i] == mask_token:
                outstr += self.tokenizer.inv_vocab[trans[i][0]] + " "
            else:
                outstr += v_tokens[i] 
        print("bokete = ", outstr)

def _load_data_using_dataset_api(input_file):
    return PickleDataset(open(input_file, 'rb'))

    
def push_test(pickle_name, tok_model_file, tok_vocab_file, bert_model_file, config_file):
    dataset = _load_data_using_dataset_api(pickle_name)
    converter = Converter()
    data = [dataset.get_example(0)]
    data_conv = converter(data, 0)
    bert_config = modeling.BertConfig.from_json_file(config_file)
    bert = BertModel(config=bert_config)
    bert_pre = BertPretrainer(bert)
    bert_pre.to_gpu()
    serializers.load_npz(bert_model_file, bert_pre)
    # print(bert_pre(data_conv[0],data_conv[1],data_conv[2],data_conv[3],data_conv))
        
def main(tok_model_file, tok_vocab_file, bert_model_file, config_file, text):

    # push_test(tok_model_file)
    
    bert_config = modeling.BertConfig.from_json_file(config_file)
    tokenizer = FullTokenizer(model_file=tok_model_file, vocab_file=tok_vocab_file, do_lower_case=False)
    bert = BertModel(config=bert_config)
    bert_pre = BertPretrainer(bert)
    interp = BertInterpolater(bert_pre, tokenizer)
    interp.to_gpu()
    serializers.load_npz(bert_model_file, bert_pre)
    interp.push_text(text)
    
if __name__=='__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
