# encoding=utf8

import paddle.fluid as fluid
import collections
import os
from paddle.fluid import layers


def compute_loss(output_tensors, args=None):
    """Compute loss for mrc model"""
    def _compute_single_loss(logits, positions):
        """Compute start/end loss for mrc model"""
        loss = fluid.layers.softmax_with_cross_entropy(
            logits=logits, label=positions)
        loss = fluid.layers.mean(x=loss)
        return loss

    start_logits = output_tensors['start_logits']
    end_logits = output_tensors['end_logits']
    start_positions = output_tensors['start_positions']
    end_positions = output_tensors['end_positions']
    start_loss = _compute_single_loss(start_logits, start_positions)
    end_loss = _compute_single_loss(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2.0
    if args.use_fp16 and args.loss_scaling > 1.0:
        total_loss = total_loss * args.loss_scaling

    return total_loss


def create_model(reader_input, base_model=None, is_training=True, args=None):
    """
        given the base model, reader_input
        return the output tensors
    """

    if is_training:
        src_ids, pos_ids, sent_ids, input_mask, start_positions, end_positions = reader_input
    else:
        src_ids, pos_ids, sent_ids, input_mask, unique_id = reader_input

    enc_out = base_model.final_word_representation
    logits = fluid.layers.fc(
        input=enc_out,
        size=2,
        num_flatten_dims=2,
        param_attr=fluid.ParamAttr(
            name="cls_squad_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_squad_out_b", initializer=fluid.initializer.Constant(0.)))

    logits = fluid.layers.transpose(x=logits, perm=[2, 0, 1])
    start_logits, end_logits = fluid.layers.unstack(x=logits, axis=0)

    batch_ones = fluid.layers.fill_constant_batch_size_like(
        input=start_logits, dtype='int64', shape=[1], value=1)
    num_seqs = fluid.layers.reduce_sum(input=batch_ones)

    output_tensors = {}
    output_tensors['start_logits'] = start_logits
    output_tensors['end_logits'] = end_logits
    output_tensors['num_seqs'] = num_seqs
    if is_training:
        output_tensors['start_positions'] = start_positions
        output_tensors['end_positions'] = end_positions

    else:
        output_tensors['unique_id'] = unique_id

    return output_tensors


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def postprocess(fetch_results):
    np_unique_ids= fetch_results['unique_id']
    np_start_logits= fetch_results['start_logits']
    np_end_logits= fetch_results['end_logits']
    ret = []
    for idx in range(np_unique_ids.shape[0]):                                                                                                                                          
        if np_unique_ids[idx] < 0:
            continue
        unique_id = int(np_unique_ids[idx])
        start_logits = [float(x) for x in np_start_logits[idx].flat]
        end_logits = [float(x) for x in np_end_logits[idx].flat]
        ret.append(
            RawResult(
                unique_id=unique_id,
                start_logits=start_logits,
                end_logits=end_logits))
    return ret

    
def global_postprocess(pred_buf, processor, mtl_args, task_args):
    if not os.path.exists(mtl_args.checkpoint_path):
        os.makedirs(mtl_args.checkpoints)
    output_prediction_file = os.path.join(mtl_args.checkpoint_path, "predictions.json")
    output_nbest_file = os.path.join(mtl_args.checkpoint_path, "nbest_predictions.json")
    output_null_log_odds_file = os.path.join(mtl_args.checkpoint_path, "null_odds.json")

    processor.write_predictions(pred_buf, task_args.n_best_size, task_args.max_answer_length,
                                task_args.do_lower_case, output_prediction_file,
                                output_nbest_file, output_null_log_odds_file,
                                task_args.with_negative,
                                task_args.null_score_diff_threshold, task_args.verbose)


