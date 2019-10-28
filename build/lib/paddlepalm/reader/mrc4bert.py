# -*- coding: UTF-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from paddlepalm.interface import reader
from paddlepalm.utils.textprocess_helper import is_whitespace
from paddlepalm.reader.utils.mrqa_helper import MRQAExample, MRQAFeature
import paddlepalm.tokenizer.bert_tokenizer as tokenization

class Reader(reader):
    
    def __init__(self, config, phase='train', dev_count=1, print_prefix=''):
        """
        Args:
            phase: train, eval, pred
            """

        self._is_training = phase == 'train'

        self._tokenizer = tokenization.FullTokenizer(
            vocab_file=config['vocab_path'], do_lower_case=config.get('do_lower_case', False))
        self._max_seq_length = config['max_seq_len']
        self._doc_stride = config['doc_stride']
        self._max_query_length = config['max_query_len']

        if phase == 'train':
            self._input_file = config['train_file']
            self._num_epochs = config['num_epochs']
            self._shuffle = config.get('shuffle', False)
            self._shuffle_buffer = config.get('shuffle_buffer', 5000)
        if phase == 'eval':
            self._input_file = config['dev_file']
            self._num_epochs = 1
            self._shuffle = False
        elif phase == 'pred':
            self._input_file = config['predict_file']
            self._num_epochs = 1
            self._shuffle = False

        # self._batch_size = 
        self._batch_size = config['batch_size']
        self._pred_batch_size = config.get('pred_batch_size', self._batch_size)
        self._print_first_n = config.get('print_first_n', 1)
        self._with_negative = config.get('with_negative', False)
        self._sample_rate = config.get('sample_rate', 0.02)

        # TODO: without slide window version
        self._with_slide_window = config.get('with_slide_window', False)

        self.vocab = self._tokenizer.vocab
        self.vocab_size = len(self.vocab)
        self.pad_id = self.vocab["[PAD]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.mask_id = self.vocab["[MASK]"]

        self.current_train_example = -1
        self.num_train_examples = -1
        self.current_train_epoch = -1

        self.n_examples = None

        print(print_prefix + 'reading raw data...')
        with open(input_file, "r") as reader:
            self.raw_data = json.load(reader)["data"]
        print(print_prfix + 'done!')

    @property
    def outputs_attr(self):
        if self._is_training:
            return {"token_ids": [[-1, self.max_seq_len, 1], 'int64'],
                    "position_ids": [[-1, self.max_seq_len, 1], 'int64'],
                    "segment_ids": [[-1, self.max_seq_len, 1], 'int64'],
                    "input_mask": [[-1, self.max_seq_len, 1], 'float32'],
                    "start_positions": [[-1, self.max_seq_len, 1], 'int64'],
                    "end_positions": [[-1, self.max_seq_len, 1], 'int64']
                    }
        else:
            return {"token_ids": [[-1, self.max_seq_len, 1], 'int64'],
                    "position_ids": [[-1, self.max_seq_len, 1], 'int64'],
                    "segment_ids": [[-1, self.max_seq_len, 1], 'int64'],
                    "input_mask": [[-1, self.max_seq_len, 1], 'float32'],
                    "unique_ids": [[-1, 1], 'int64']
                    }

    def iterator(self): 

        features = []
        for i in self._num_epochs:
            if self._is_training:
                print(self.print_prefix + '{} epoch {} {}'.format('-'*16, i, '-'*16))
            example_id = 0
            feature_id = 1000000000
            for line in self.train_file:
                raw = self.parse_line(line)

                examples = _raw_to_examples(raw['context'], raw['qa_list'], is_training=self._is_training)
                for example in examples:
                    features.extend(_example_to_features(example, example_id, self._tokenizer, \
                                        self._max_seq_length, self._doc_stride, self._max_query_length, \
                                        id_offset=1000000000+len(features), is_training=self._is_training))
                    if len(features) >= self._batch_size * self._dev_count:
                        for batch, total_token_num in _features_to_batches( \
                                                        features[:self._batch_size * self._dev_count], \
                                                        batch_size, in_tokens=self._in_tokens):
                            temp = prepare_batch_data(batch, total_token_num, \
                                    max_len=self._max_seq_length, voc_size=-1, \
                                    pad_id=self.pad_id, cls_id=self.cls_id, sep_id=self.sep_id, mask_id=-1, \
                                    return_input_mask=True, return_max_len=False, return_num_token=False)
                            if self._is_training:
                                tok_ids, pos_ids, seg_ids, input_mask, start_positions, end_positions = temp
                                yield {"token_ids": tok_ids, "position_ids": pos_ids, "segment_ids": seg_ids, "input_mask": input_mask, "start_positions": start_positions, 'end_positions': end_positions}
                            else:
                                tok_ids, pos_ids, seg_ids, input_mask, unique_ids = temp
                                yield {"token_ids": tok_ids, "position_ids": pos_ids, "segment_ids": seg_ids, "input_mask": input_mask, "unique_ids": unique_ids}
                                
                        features = features[self._batch_size * self._dev_count:]
                    example_id += 1

        # The last batch may be discarded when running with distributed prediction, so we build some fake batches for the last prediction step.
        if self._is_training and len(features) > 0:
            pred_batches = []
            for batch, total_token_num in _features_to_batches( \
                                            features[:self._batch_size * self._dev_count], \
                                            batch_size, in_tokens=self._in_tokens):
                pred_batches.append(prepare_batch_data(batch, total_token_num, max_len=self._max_seq_length, voc_size=-1,
                                        pad_id=self.pad_id, cls_id=self.cls_id, sep_id=self.sep_id, mask_id=-1, \
                                        return_input_mask=True, return_max_len=False, return_num_token=False))

            fake_batch = pred_batches[-1]
            fake_batch = fake_batch[:-1] + [np.array([-1]*len(fake_batch[0]))]
            pred_batches = pred_batches + [fake_batch] * (dev_count - len(pred_batches))
            for batch in pred_batches:
                yield batch

    @property
    def num_examples(self):
        if self.n_examples is None:
            self.n_examples = _estimate_runtime_examples(self.raw_data, self._sample_rate, self._tokenizer, \
                                  self._max_seq_length, self._doc_stride, self._max_query_length, \
                                  remove_impossible_questions=True, filter_invalid_spans=True)
        return self.n_examples
        # return math.ceil(n_examples * self._num_epochs / float(self._batch_size * self._dev_count))



def _raw_to_examples(context, qa_list, is_training=True, remove_impossible_questions=True, filter_invalid_spans=True):
    """
    Args:
        context: (str) the paragraph that provide information for QA
        qa_list: (list) nested dict. Each element in qa_list should contain at least 'id' and 'question'. And the ....
        """
    examples = []
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in context:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    for qa in qa_list:
        qas_id = qa["id"]
        question_text = qa["question"]
        start_position = None
        end_position = None
        orig_answer_text = None
        is_impossible = False
        if is_training:

            assert len(qa["answers"]) == 1, "For training, each question should have exactly 1 answer."

            if ('is_impossible' in qa) and (qa["is_impossible"]):
                if remove_impossible_questions or filter_invalid_spans:
                    continue
                else:
                    start_position = -1
                    end_position = -1
                    orig_answer_text = ""
                    is_impossible = True
            else:
                answer = qa["answers"][0]
                orig_answer_text = answer["text"]
                answer_offset = answer["answer_start"]
                answer_length = len(orig_answer_text)
                start_position = char_to_word_offset[answer_offset]
                end_position = char_to_word_offset[answer_offset +
                                                   answer_length - 1]

                # remove corrupt samples
                actual_text = " ".join(doc_tokens[start_position:(
                    end_position + 1)])
                cleaned_answer_text = " ".join(
                    tokenization.whitespace_tokenize(orig_answer_text))
                if actual_text.find(cleaned_answer_text) == -1:
                    print(self.print_prefix + "Could not find answer: '%s' vs. '%s'",
                          actual_text, cleaned_answer_text)
                    continue

        examples.append(MRQAExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible))

    return examples




def _example_to_features(example, example_id, tokenizer, max_seq_length, doc_stride, max_query_length, id_offset, is_training):

    query_tokens = tokenizer.tokenize(example.question_text)

    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens = []
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    tok_start_position = None
    tok_end_position = None
    if is_training and example.is_impossible:
        tok_start_position = -1
        tok_end_position = -1
    if is_training and not example.is_impossible:
        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position +
                                                 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

    # We can have documents that are longer than the maximum sequence length.
    # To deal with this we do a sliding window approach, where we take chunks
    # of the up to our max length with a stride of `doc_stride`.
    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        if length > max_tokens_for_doc:
            length = max_tokens_for_doc
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, doc_stride)

    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        token_is_max_context = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[
                split_token_index]

            is_max_context = _check_is_max_context(
                doc_spans, doc_span_index, split_token_index)
            token_is_max_context[len(tokens)] = is_max_context
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        #while len(input_ids) < max_seq_length:
        #  input_ids.append(0)
        #  input_mask.append(0)
        #  segment_ids.append(0)

        #assert len(input_ids) == max_seq_length
        #assert len(input_mask) == max_seq_length
        #assert len(segment_ids) == max_seq_length

        start_position = None
        end_position = None
        if is_training and not example.is_impossible:
            # For training, if our document chunk does not contain an annotation
            # we throw it out, since there is nothing to predict.
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            out_of_span = False
            if not (tok_start_position >= doc_start and
                    tok_end_position <= doc_end):
                out_of_span = True
            if out_of_span:
                start_position = 0
                end_position = 0
                continue
            else:
                doc_offset = len(query_tokens) + 2
                start_position = tok_start_position - doc_start + doc_offset
                end_position = tok_end_position - doc_start + doc_offset

        if is_training and example.is_impossible:
            start_position = 0
            end_position = 0
        
        def format_print():
            print("*** Example ***")
            print("unique_id: %s" % (unique_id))
            print("example_index: %s" % (example_index))
            print("doc_span_index: %s" % (doc_span_index))
            print("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            print("token_to_orig_map: %s" % " ".join([
                "%d:%d" % (x, y)
                for (x, y) in six.iteritems(token_to_orig_map)
            ]))
            print("token_is_max_context: %s" % " ".join([
                "%d:%s" % (x, y)
                for (x, y) in six.iteritems(token_is_max_context)
            ]))
            print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            print("segment_ids: %s" %
                  " ".join([str(x) for x in segment_ids]))
            if is_training and example.is_impossible:
                print("impossible example")
            if is_training and not example.is_impossible:
                answer_text = " ".join(tokens[start_position:(end_position +
                                                              1)])
                print("start_position: %d" % (start_position))
                print("end_position: %d" % (end_position))
                print("answer: %s" %
                      (tokenization.printable_text(answer_text)))

        if self._print_first_n > 0:
            format_print()
            self._print_first_n -= 1

        features.append(MRQAFeature(
            unique_id=id_offset,
            example_index=example_id,
            doc_span_index=doc_span_index,
            tokens=tokens,
            token_to_orig_map=token_to_orig_map,
            token_is_max_context=token_is_max_context,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position,
            is_impossible=example.is_impossible))

        id_offset += 1

    return features


def _features_to_batches(features, batch_size, in_tokens):
    batch, total_token_num, max_len = [], 0, 0
    for (index, feature) in enumerate(features):
        if phase == 'train':
            self.current_train_example = index + 1
        seq_len = len(feature.input_ids)
        labels = [feature.unique_id
                  ] if feature.start_position is None else [
                      feature.start_position, feature.end_position
                  ]
        example = [
            feature.input_ids, feature.segment_ids, range(seq_len)
        ] + labels
        max_len = max(max_len, seq_len)

        if in_tokens:
            to_append = (len(batch) + 1) * max_len <= batch_size
        else:
            to_append = len(batch) < batch_size

        if to_append:
            batch.append(example)
            total_token_num += seq_len
        else:
            yield batch, total_token_num
            batch, total_token_num, max_len = [example
                                               ], seq_len, seq_len
    if len(batch) > 0:
        yield batch, total_token_num


def _estimate_runtime_examples(data, sample_rate, tokenizer, \
                              max_seq_length, doc_stride, max_query_length, \
                              remove_impossible_questions=True, filter_invalid_spans=True):
    """Count runtime examples which may differ from number of raw samples due to sliding window operation and etc.. 
       This is useful to get correct warmup steps for training."""

    assert sample_rate > 0.0 and sample_rate <= 1.0, "sample_rate must be set between 0.0~1.0"

    num_raw_examples = 0
    for entry in data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
            for qa in paragraph["qas"]:
                num_raw_examples += 1
    # print("num raw examples:{}".format(num_raw_examples))

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    sampled_examples = []
    first_samp = True
    for entry in data:
        for paragraph in entry["paragraphs"]:
            doc_tokens = None
            for qa in paragraph["qas"]:
                if not first_samp and random.random() > sample_rate and sample_rate < 1.0:
                    continue

                if doc_tokens is None:
                    paragraph_text = paragraph["context"]
                    doc_tokens = []
                    char_to_word_offset = []
                    prev_is_whitespace = True
                    for c in paragraph_text:
                        if is_whitespace(c):
                            prev_is_whitespace = True
                        else:
                            if prev_is_whitespace:
                                doc_tokens.append(c)
                            else:
                                doc_tokens[-1] += c
                            prev_is_whitespace = False
                        char_to_word_offset.append(len(doc_tokens) - 1)

                assert len(qa["answers"]) == 1, "For training, each question should have exactly 1 answer."

                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False

                if ('is_impossible' in qa) and (qa["is_impossible"]):
                    if remove_impossible_questions or filter_invalid_spans:
                        continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""
                        is_impossible = True
                else:
                    answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)
                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset +
                                                       answer_length - 1]

                    # remove corrupt samples
                    actual_text = " ".join(doc_tokens[start_position:(
                        end_position + 1)])
                    cleaned_answer_text = " ".join(
                        tokenization.whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        continue

                example = MRQAExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)

                sampled_examples.append(example)
                first_samp = False

    
    runtime_sample_rate = len(sampled_examples) / float(num_raw_examples)

    runtime_samp_cnt = 0

    for example in sampled_examples:
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None

        tok_start_position = orig_to_tok_index[example.start_position]
        if example.end_position < len(example.doc_tokens) - 1:
            tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
        else:
            tok_end_position = len(all_doc_tokens) - 1
        (tok_start_position, tok_end_position) = _improve_answer_span(
            all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            doc_start = doc_span.start
            doc_end = doc_span.start + doc_span.length - 1
            if filter_invalid_spans and not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                continue
            runtime_samp_cnt += 1
    return int(runtime_samp_cnt/runtime_sample_rate)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The MRQA annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in MRQA, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context,
                    num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index

