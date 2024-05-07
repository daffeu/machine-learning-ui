from keras.optimizers.optimizer import Optimizer
import tensorflow as tf
import numpy as np


class Apollo(Optimizer):
    """Optimizer that implements the Apollo algorithm.

    Apollo is a nonconvex stochastic optimization method
    that incorporates the curvature of the loss function by
    approximating the Hessian using a diagonal matrix.
    It is a quasi-Newton approach.
    Reference:
    - [Xuezhe Ma, 2020] <https://arxiv.org/abs/2009.13586>_

    Parameters
    ----------
    learning_rate : float, default: 1e-2
        The learning rate.
    beta : float, default: 0.9
        The coefficient used for computing running averages of gradient.
    epsilon : float, default: 1e-4
        The term added to the denominator to improve numerical stability.
    rebound : str, default: “constant”
        The recified bound for diagonal hessian: “constant” | “belief”.
    warmup : int, default: 500
        The number of warmup steps.
    init_lr : float, default: 1e-5
        The initial learning rate for warmup.
    weight_decay : float, default: 0.0
        The weight decay coefficient.
    """
    def __init__(
            self,
            learning_rate=0.01,
            beta=0.9,
            epsilon=1e-4,
            init_lr=None,
            warmup=500,
            rebound="constant",
            weight_decay=0.0,
            clipnorm=None,
            clipvalue=None,
            global_clipnorm=None,
            use_ema=False,
            ema_momentum=0.99,
            ema_overwrite_frequency=None,
            jit_compile=True,
            name="Apollo",
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

        if init_lr is None:
            init_lr=learning_rate/1000

        self._learning_rate = self._build_learning_rate(learning_rate)
        self.beta = beta
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.init_lr = init_lr
        self.warmup = warmup
        self.rebound_type = rebound

    def build(self, var_list):
        """Initialize optimizer variables.

        Apollo optimizer has 3 types of variables: exponential moving average
        of gradient values, hessian approximation values
        and previous update direction.

        Parameters
        ----------
        var_list
            List of model variables to build Apollo variables on.
        """

        super().build(var_list)

        if hasattr(self, "_built") and self._built:
            return

        self._built = True
        self._exp_avg_grads = []
        self._approx_hessians = []
        self._updates = []
        for var in var_list:
            self._exp_avg_grads.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="m"
                )
            )

            self._approx_hessians.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="B"
                )
            )

            self._updates.append(
                self.add_variable_from_reference(
                    model_variable=var, variable_name="d_p"
                )
            )

    @tf.function
    def update_step(self, gradient, variable):
        """Update step given gradient and the associated model variable."""

        curr_lr = tf.cast(self.learning_rate, variable.dtype)
        weight_decay = tf.cast(self.weight_decay, variable.dtype)
        beta = tf.cast(self.beta, variable.dtype)
        eps = tf.cast(self.epsilon, variable.dtype)
        warmup = tf.cast(self.warmup, variable.dtype)
        init_lr_t = tf.cast(self.init_lr, variable.dtype)
        step = tf.cast(self.iterations + 1, variable.dtype)

        var_key = self._var_key(variable)
        m = self._exp_avg_grads[self._index_dict[var_key]]
        B = self._approx_hessians[self._index_dict[var_key]]
        d_p = self._updates[self._index_dict[var_key]]

        # Calculate current lr
        if step <= warmup:
            curr_lr = init_lr_t + (curr_lr - init_lr_t) * step / warmup

        bias_correction = 1 - beta ** step
        alpha = (1 - beta) / bias_correction

        delta_grad = gradient - m
        if self.rebound_type == "belief":
            rebound = tf.norm(delta_grad, ord=np.inf)
        else:
            rebound = 0.01
            eps = eps / rebound

        m.assign_add(delta_grad * alpha)

        denom = tf.norm(d_p, ord=4) + eps
        d_p_t = tf.divide(d_p, denom)
        v_sq = tf.multiply(d_p_t, d_p_t)
        delta = tf.math.reduce_sum(tf.multiply(tf.divide(d_p, denom), d_p_t)) *\
                (-alpha) - tf.math.reduce_sum(tf.multiply(B, v_sq))

        B.assign_add(v_sq * delta)

        if self.rebound_type == 'belief':
            denom = tf.math.maximum(tf.math.abs(B), rebound) + tf.divide(eps, alpha)
        else:
            denom = tf.math.maximum(tf.math.abs(B), rebound)

        d_p_t = tf.divide(m, denom)

        if weight_decay != 0:
            d_p_t = d_p.assign(d_p_t + variable * weight_decay)
        else:
            d_p_t = d_p.assign(d_p_t)

        variable.assign_add(d_p_t * -curr_lr)

    def get_config(self):
        config = super().get_config()

        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta": self._serialize_hyperparameter("beta"),
                "epsilon": self.epsilon,
                "rebound": self.rebound_type,
                "weight_decay": self._serialize_hyperparameter("weight_decay"),

            }
        )
        return config
