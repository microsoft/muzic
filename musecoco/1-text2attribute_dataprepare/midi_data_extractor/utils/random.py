import numpy as np
import random


def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)