# encoding=utf8

import paddle.fluid as fluid


def compute_loss(output_tensors, args=None):
    """Compute loss for mrc model"""
    labels = output_tensors['labels']
    logits = output_tensors['logits']

    ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
        logits=logits, label=labels, return_softmax=True)
    loss = fluid.layers.mean(x=ce_loss)

    if args.use_fp16 and args.loss_scaling > 1.0:
        loss *= args.loss_scaling

    return loss


def create_model(reader_input, base_model=None, is_training=True, args=None):
    """
        given the base model, reader_input
        return the output tensors
    """
    labels = reader_input[-1]

    cls_feats = base_model.final_sentence_representation
    cls_feats = fluid.layers.dropout(
        x=cls_feats,
        dropout_prob=0.1,
        dropout_implementation="upscale_in_train")
    logits = fluid.layers.fc(
        input=cls_feats,
        size=2,
        param_attr=fluid.ParamAttr(
            name="cls_out_w",
            initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
            name="cls_out_b", initializer=fluid.initializer.Constant(0.)))

    num_seqs = fluid.layers.fill_constant(shape=[1], value=512, dtype='int64') 

    output_tensors = {}
    output_tensors['labels'] = labels
    output_tensors['logits'] = logits
    output_tensors['num_seqs'] = num_seqs

    return output_tensors
