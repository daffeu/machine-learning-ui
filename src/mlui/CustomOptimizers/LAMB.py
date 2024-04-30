from keras.optimizers.optimizer import Optimizer
import tensorflow as tf
import numpy as np


class LAMB(Optimizer):
    """Optimizer that implements the LAMB algorithm.

    Layer-wise Adaptive Large Batch Optimization Technique, is an optimization algorithm widely used in deep learning.
    The algorithm is designed for training deep neural networks and shows good convergence.
    LAMB combines adaptability using the first and second moments, and layer-by-layer scaling.
    Reference:
    - [Yang You, Jing Li, Sashank Reddi, Jonathan Hseu, Sanjiv Kumar, Srinadh Bhojanapalli,
        Xiaodan Song, James Demmel, Kurt Keutzer, Cho-Jui Hsieh, 2020] <https://arxiv.org/abs/1904.00962>_

    Parameters
    ----------
    learning_rate : float, default: 1e-3
       The learning rate.
    beta_1 : float, default: 0.9
       The exponential decay rate for the 1st moment estimates.
    beta_2 : float, default: 0.0
       The exponential decay rate for the 2nd moment estimates.
    epsilon : float, default: 1e-8
       The term added to the denominator to improve numerical stability.
    weight_decay : float, default: 0.0
       The weight decay.
    """

    def __init__(
            self,
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-6,
            weight_decay=0.0,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="LAMB",
            **kwargs
    ):
        super().__init__(
            name=name,
            clipnorm=clipnorm,
            clipvalue=clipvalue,
            global_clipnorm=global_clipnorm,
            use_ema=use_ema,
            ema_momentum=ema_momentum,
            ema_overwrite_frequency=ema_overwrite_frequency,
            jit_compile=jit_compile,
            **kwargs
        )

        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.epsilon = epsilon

    def build(self, var_list):
        """Initialize optimizer variables.

        LAMB optimizer has 2 types of variables: exponential moving average
        of gradient values, exponential moving average of squared gradient
        values.

        Parameters
        ----------
        var_list
            List of model variables to build LAMB variables on.
        """
        super().build(var_list)
        if hasattr(self, "_built") and self._built:
            return
        self.m = []
        self.v = []
        for var in var_list:
            self.m.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )
            self.v.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="v"
                )
            )
        self._built = True

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""
        lr = tf.cast(self.learning_rate, variable.dtype)
        local_step = tf.cast(self.iterations + 1, variable.dtype)
        eps = tf.cast(self.epsilon, variable.dtype)
        beta_1_power = tf.pow(tf.cast(self.beta_1, variable.dtype), local_step)
        beta_2_power = tf.pow(tf.cast(self.beta_2, variable.dtype), local_step)

        var_key = self._var_key(variable)
        m = self.m[self._index_dict[var_key]]
        v = self.v[self._index_dict[var_key]]

        m_t = m.assign(self.beta_1*m + (1 - self.beta_1)*gradient)
        v_t = v.assign(self.beta_2*v + (1 - self.beta_2)*tf.pow(gradient, 2))

        m_t = m_t/(1 - beta_1_power)
        v_t = v_t/(1 - beta_2_power)

        r_t = m_t/(tf.sqrt(v_t)+eps)
        if self.weight_decay != 0:
            r_t += self.weight_decay*variable

        w_norm = tf.norm(variable, ord=2)
        g_norm = tf.norm(variable, ord=2)

        ratio = tf.where(
            tf.greater(w_norm, 0),
            tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0),
            1.0,
        )

        variable.assign_add(-r_t*lr*ratio)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "epsilon": self.epsilon,
                "weight_decay": self._serialize_hyperparameter("weight_decay"),

            }
        )
        return config




