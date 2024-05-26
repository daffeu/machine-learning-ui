from keras.optimizers.optimizer import Optimizer
import tensorflow as tf


class MADGRAD(Optimizer):
    """Optimizer that implements the MADGRAD algorithm.

    MADGRAD is an optimization algorithm used in deep learning.
    It combines adaptive learning rates and momentum to efficiently
    navigate the loss landscape during training.
    With its ability to dynamically adjust the learning rate and smooth
    out oscillations using momentum,
    MADGRAD offers fast convergence and improved optimization performance.
    Reference:
    - [Aaron Defazio, Samy Jelassi, 2021] <https://arxiv.org/abs/2101.11075>_

    Parameters
    ----------
    learning_rate : float, default: 1e-2
        The learning rate.
    momentum : float, default: 0.0
        The momentum value in the range [0,1).
    weight_decay : float, default: 0.0
        The weight decay (L2 penalty).
    epsilon : float, default: 1e-6
        The term added to the denominator to improve numerical stability.
    """

    def __init__(
            self,
            learning_rate=0.01,
            momentum=0.9,
            epsilon=1e-6,
            weight_decay=None,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="MADGRAD",
            **kwargs
    ):
        super().__init__(
            name=name,
            weight_decay=weight_decay,
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
        self.momentum = momentum
        self.epsilon = epsilon
        if momentum != 0:
            self.use_momentum = True

    def build(self, var_list):
        """Initialize optimizer variables.

        MADGRAD optimizer has 2 types of variables: exponential average
        of gradient values, exponential average of squared gradient.

        Parameters
        ----------
        var_list
            List of model variables to build Apollo variables on.
        """

        super().build(var_list)

        if hasattr(self, "_built") and self._built:
            return

        self._built = True
        self.m = []
        self.v = []
        self.x0 = []
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
            self.x0.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="x0"
                )
            )

    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""

        lr_t = tf.cast(self.learning_rate, variable.dtype)
        momentum = tf.cast(self.momentum, variable.dtype)
        eps = tf.cast(self.epsilon, variable.dtype)
        step = tf.cast(self.iterations + 1, variable.dtype)

        var_key = self._var_key(variable)
        m = self.m[self._index_dict[var_key]]
        v = self.v[self._index_dict[var_key]]
        x0 = self.x0[self._index_dict[var_key]]
        lamb = lr_t * tf.sqrt(step)

        if isinstance(gradient, tf.IndexedSlices):
            # Sparse gradients.
            m.assign_add(m)
            m.scatter_add(
                tf.IndexedSlices(
                    gradient.values * lamb, gradient.indices
                )
            )

            v.assign_add(v)
            v.scatter_add(
                tf.IndexedSlices(
                    tf.square(gradient.values) * lamb, gradient.indices
                )
            )

            denom = tf.pow(v, 1 / 3) + eps

            z = x0 - tf.divide(m, denom)

            var = tf.multiply((1 - momentum), variable) + tf.multiply(momentum, z)

            variable.assign(var)

        else:
            # Dence gradients.
            m.assign_add(lamb * gradient)
            v.assign_add(lamb * (gradient * gradient))

            denom = tf.pow(v, 1 / 3) + eps

            z = x0 - tf.divide(m, denom)

            var = tf.multiply((1 - momentum), variable) + tf.multiply(momentum, z)

            variable.assign(var)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                'learning_rate': self._serialize_hyperparameter(
                    self._learning_rate
                ),
                'momentum': self.momentum,
                'epsilon': self.epsilon,
            }
        )
        return config
