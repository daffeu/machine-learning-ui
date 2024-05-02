import tensorflow as tf
from mlui.CustomOptimizers import AdaMod, Apollo, MADGRAD, LARS, AdaHessian, LAMB, RAdam

import mlui.types.classes as ct
import mlui.types.widgets as wt
import mlui.widgets.optimizers as widget

classes: ct.OptimizerTypes = {
    "Adam": tf.keras.optimizers.Adam,
    "RMSprop": tf.keras.optimizers.RMSprop,
    "SGD": tf.keras.optimizers.SGD,
    "AdaMod": AdaMod,
    "Apollo": Apollo,
    "LAMB": LAMB,
    "RAdam": RAdam,
    "MADGRAD": MADGRAD,
    "LARS": LARS,
    "AdaHessian": AdaHessian,
}

widgets: wt.OptimizerWidgetTypes = {
    "Adam": widget.Adam,
    "RMSprop": widget.RMSprop,
    "SGD": widget.SGD,
    "AdaMod": widget.AdaMod,
    "Apollo": widget.Apollo,
    "LAMB": widget.LAMB,
    "Lookahead": widget.Lookahead,
    "RAdam": widget.RAdam,
    "MADGRAD": widget.MADGRAD,
    "LARS": widget.LARS,
    "AdaHessian": widget.AdaHessian,
}
