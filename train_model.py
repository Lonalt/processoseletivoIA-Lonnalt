"""
Pipeline para treino de CNN no MNIST.
Saída: model.h5
"""

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Configuração encapsulada
def config():
    return {
        "epocas": 4,
        "batch": 32,
        "val": 0.2,
        "arquivo": "model.h5",
        "seed": 7,
    }

