"""

Author: Sayed Kamaledin Ghiasi-Shirazi
Year:     2021

Associated Publication:
    Ghiasi-Shirazi, K. Competitive Cross-Entropy Loss: A Study on Training Single-Layer Neural Networks for 
                       Solving Nonlinearly Separable Classification Problems. 
                       Neural Process Lett 50, 1115–1122 (2019). 
                       https://doi.org/10.1007/s11063-018-9906-5

"""

import tensorflow as tf
from tensorflow.python.keras import backend as backend
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.util import dispatch
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.metrics import MeanMetricWrapper

@keras_export('keras.metrics.sparse_competitive_crossentropy',
              'keras.losses.sparse_competitive_crossentropy')
@dispatch.add_dispatch_support
def sparse_competitive_crossentropy(y_true_label,
                              y_pred,
                              C,
                              K):
    y_true_label = tf.reshape(y_true_label, [-1])
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = tf.one_hot (y_true_label, C)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    y_true = tf.repeat(y_true, repeats=K, axis = 1)
    y_pred2 = tf.stop_gradient (y_pred)
    y_pred2_max = tf.reduce_max(y_pred2, axis=1, keepdims=True)
    y_pred2 = y_pred2 - y_pred2_max
    y_true = tf.multiply(y_true, tf.exp(y_pred2))
    y_true = tf.linalg.normalize(y_true, axis=1,ord=1)[0]
    y_true = tf.stop_gradient (y_true)
    return backend.categorical_crossentropy(y_true, y_pred, from_logits=True)

@keras_export('keras.losses.SparseCompetitiveCrossentropy')
class SparseCompetitiveCrossentropy(LossFunctionWrapper):
    def __init__(self,
                 C,
                 K,
                 reduction=losses_utils.ReductionV2.AUTO,
                 name='sparse_competitive_crossentropy'):
        super(SparseCompetitiveCrossentropy, self).__init__(
                sparse_competitive_crossentropy,
                name=name,
                reduction=reduction,
                C=C,
                K=K)

@keras_export('keras.metrics.multi_prototype_sparse_categorical_accuracy')
@dispatch.add_dispatch_support
def multi_prototype_sparse_categorical_accuracy(y_true, y_pred, K):
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = ops.convert_to_tensor_v2_with_dispatch(y_true)
    if y_true.shape.ndims > 1:
        y_true = array_ops.squeeze(y_true, axis=-1)
    y_pred = math_ops.argmax(y_pred, axis=-1)

    # If the predicted output and actual output types don't match, force cast them
    # to match.
    if backend.dtype(y_pred) != backend.dtype(y_true):
      y_pred = math_ops.cast(y_pred, backend.dtype(y_true))
    return math_ops.cast(math_ops.equal(y_true, y_pred//K), backend.floatx())


@keras_export('keras.metrics.MultiprototypeSparseCategoricalAccuracy')
class MultiprototypeSparseCategoricalAccuracy(MeanMetricWrapper):
    def __init__(self, K, name='multi_prototype_sparse_categorical_accuracy', dtype=None):
        super(MultiprototypeSparseCategoricalAccuracy, self).__init__(
        multi_prototype_sparse_categorical_accuracy, name, dtype=dtype, K=K)


