import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

delta = 1.0


def jrelu(x):
    # return tf.cond(x > tf.constant(delta), lambda: x, lambda: tf.constant(0.0))
    if (x > delta) is not None:
        return x
    else:
        return 0.0


def jrelu_grad(x):
    # print('gard:', tf.cond(x > tf.constant(delta), lambda: tf.constant(1.0), lambda: tf.constant(0.0)))
    # l = tf.cond(x > tf.constant(delta), lambda: 1.0, lambda: 0.0)
    # return l
    if (x > delta) is not None:
        return 1.0
    else:
        return 0.0


jrelu_np = np.vectorize(jrelu)
jrelu_grad_np = np.vectorize(jrelu_grad)

jrelu_np_32 = lambda x: jrelu_np(x).astype(x.dtype)
jrelu_grad_np_32 = lambda x: jrelu_grad_np(x).astype(x.dtype)


def jrelu_grad_tf(x, name=None):
    with ops.name_scope(name, "Relu_grad", [x]) as name:
        y = tf.py_func(jrelu_grad_np_32, [x], [x.dtype], name=name, stateful=False)
        # y[0].set_shape(0)
        # print(y[0])
        return y[0]


def my_py_func(func, inp, Tout, stateful=False, name=None, my_grad_func=None):
    # need to generate a unique name to avoid duplicates:
    random_name = "PyFuncGrad" + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(random_name)(my_grad_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": random_name, "PyFuncStateless": random_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


def _jrelu_grad(op, pred_grad):
    x = op.inputs[0]
    cur_grad = jrelu_grad(x)
    next_grad = pred_grad * cur_grad
    return next_grad

def jrelu_tf(x, name=None):
    with ops.name_scope(name, "Relu", [x]) as name:
        y = my_py_func(jrelu_np_32,
                       [x],
                       [x.dtype],
                       stateful=True,
                       name=name,
                       my_grad_func=_jrelu_grad)
    y[0].set_shape(x.shape)
    return y[0]