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

import paddle.fluid as fluid
from paddlepalm.head.base_head import Head
import collections
import numpy as np
import os
import math
import six
import paddlepalm.tokenizer.ernie_tokenizer as tokenization
import json
import io

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

class MRC(Head):
    """
    Machine Reading Comprehension
    """

    def __init__(self, max_query_len, input_dim, pred_output_path=None, verbose=False, with_negative=False, do_lower_case=False, max_ans_len=None, null_score_diff_threshold=0.0, n_best_size=20, phase='train'):

        self._is_training = phase == 'train'
        self._hidden_size = input_dim
        self._max_sequence_length = max_query_len
 
        self._pred_results = []
        
        output_dir = pred_output_path
        self._max_answer_length = max_ans_len
        self._null_score_diff_threshold = null_score_diff_threshold
        self._n_best_size = n_best_size
        output_dir = pred_output_path
        self._verbose = verbose
        self._with_negative = with_negative
        self._do_lower_case = do_lower_case


    @property
    def inputs_attrs(self):
        if self._is_training:
            reader = {"start_positions": [[-1], 'int64'],
                      "end_positions": [[-1], 'int64'],
                      }
        else:
            reader = {'unique_ids': [[-1], 'int64']}
        bb = {"encoder_outputs": [[-1, -1, self._hidden_size], 'float32']}
        return {'reader': reader, 'backbone': bb}
        
    @property
    def epoch_inputs_attrs(self):
        if not self._is_training:
            from_reader = {'examples': None, 'features': None}
            return {'reader': from_reader}

    @property
    def outputs_attr(self):
        if self._is_training:
            return {'loss': [[1], 'float32']}
        else:
            return {'start_logits': [[-1, -1, 1], 'float32'],
                    'end_logits': [[-1, -1, 1], 'float32'],
                    'unique_ids': [[-1], 'int64']}


    def build(self, inputs, scope_name=""):
        if self._is_training:
            start_positions = inputs['reader']['start_positions']
            end_positions = inputs['reader']['end_positions']
            # max_position = inputs["reader"]["seqlen"] - 1
            # start_positions = fluid.layers.elementwise_min(start_positions, max_position)
            # end_positions = fluid.layers.elementwise_min(end_positions, max_position)
            start_positions.stop_gradient = True
            end_positions.stop_gradient = True
        else:
            unique_id = inputs['reader']['unique_ids']

            # It's used to help fetch variable 'unique_ids' that will be removed in the future
            helper_constant = fluid.layers.fill_constant(shape=[1], value=1, dtype='int64')
            fluid.layers.elementwise_mul(unique_id, helper_constant)  
            

        enc_out = inputs['backbone']['encoder_outputs']
        logits = fluid.layers.fc(
            input=enc_out,
            size=2,
            num_flatten_dims=2,
            param_attr=fluid.ParamAttr(
                name=scope_name+"cls_squad_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name=scope_name+"cls_squad_out_b", initializer=fluid.initializer.Constant(0.)))

        logits = fluid.layers.transpose(x=logits, perm=[2, 0, 1])
        start_logits, end_logits = fluid.layers.unstack(x=logits, axis=0)

        def _compute_single_loss(logits, positions):
            """Compute start/en
            d loss for mrc model"""
            inputs = fluid.layers.softmax(logits)
            loss = fluid.layers.cross_entropy(
                input=inputs, label=positions)
            loss = fluid.layers.mean(x=loss)
            return loss

        if self._is_training:
            start_loss = _compute_single_loss(start_logits, start_positions)
            end_loss = _compute_single_loss(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2.0
            return {'loss': total_loss}
        else:
            return {'start_logits': start_logits,
                    'end_logits': end_logits,
                    'unique_ids': unique_id}


    def batch_postprocess(self, rt_outputs):
        """this func will be called after each step(batch) of training/evaluating/predicting process."""
        if not self._is_training:
            unique_ids = rt_outputs['unique_ids']
            start_logits = rt_outputs['start_logits']
            end_logits = rt_outputs['end_logits']
            for idx in range(len(unique_ids)):
                
                if unique_ids[idx] < 0:
                    continue
                if len(self._pred_results) % 1000 == 0:
                    print("Predicting example: {}".format(len(self._pred_results)))
                uid = int(unique_ids[idx])

                s = [float(x) for x in start_logits[idx].flat]
                e = [float(x) for x in end_logits[idx].flat]
                self._pred_results.append(
                    RawResult(
                        unique_id=uid,
                        start_logits=s,
                        end_logits=e))

    def epoch_postprocess(self, post_inputs, output_dir=None):
        """(optional interface) this func will be called after evaluation/predicting process and each epoch during training process."""

        if not self._is_training:
            if output_dir is None:
                raise ValueError('argument output_dir not found in config. Please add it into config dict/file.')
            examples = post_inputs['reader']['examples']
            features = post_inputs['reader']['features']
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_prediction_file = os.path.join(output_dir, "predictions.json")
            output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
            output_null_log_odds_file = os.path.join(output_dir, "null_odds.json")
            _write_predictions(examples, features, self._pred_results,
                              self._n_best_size, self._max_answer_length,
                              self._do_lower_case, output_prediction_file,
                              output_nbest_file, output_null_log_odds_file,
                              self._with_negative,
                              self._null_score_diff_threshold, self._verbose)


def _write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file,
                      with_negative, null_score_diff_threshold,
                      verbose):
    """Write final predictions to the json file and log-odds of null if needed."""
    print("Writing predictions to: %s" % (output_prediction_file))
    print("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", [
            "feature_index", "start_index", "end_index", "start_logit",
            "end_logit"
        ])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min mull score
        ull_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
    
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[
                    0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))

        if with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1
                                                              )]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end +
                                                                 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = _get_final_text(tok_text, orig_text, do_lower_case,
                                            verbose)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))

        # if we didn't inlude the empty option in the n-best, inlcude it
        if with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(
                    text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry
        # debug
        if best_non_null_entry is None:
            print("Emmm..., sth wrong")

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text.encode('utf-8').decode('utf-8')
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text

        all_nbest_json[example.qas_id] = nbest_json
    


    with io.open(output_prediction_file, "w", encoding='utf-8') as writer:
        
        writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")

    with io.open(output_nbest_file, "w", encoding='utf-8') as writer:
        writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n")

    if with_negative:
        with io.open(output_null_log_odds_file, "w", encoding='utf-8') as writer:
            writer.write(json.dumps(scores_diff_json, indent=4, ensure_ascii=False) + "\n")


def _get_final_text(pred_text, orig_text, do_lower_case, verbose):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the MRQA eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heruistic between
    # `pred_text` and `orig_text` to get a character-to-charcter alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = tokenization.BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose:
            print("Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose:
            print("Length not equal after stripping spaces: '%s' vs '%s'",
                  orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in six.iteritems(tok_ns_to_s_map):
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose:
            print("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose:
            print("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(
        enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


