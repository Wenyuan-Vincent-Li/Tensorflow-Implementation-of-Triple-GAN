import math
import numpy as np


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    # Name the configurations. For example, 'GAN', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes

    ## Input pipeline
    DATA_NAME = None
    DATA_DIR = None  # Specify the data directory, override in sub-classes
    NUM_LABEL = None
    BATCH_SIZE = None  # Batch size
    BATCH_SIZE_L_D = None
    SAMPLE_SIZE = 64

    IMAGE_HEIGHT = None
    IMAGE_WIDTH = None
    CHANNEL = None

    REPEAT = None

    ## Model architecture
    # Number of classification classes
    Z_DIM = None
    NUM_CLASSES = None
    MINIBATCH_DIS = False

    ## Training settings
    # Restore
    RESTORE = False  # Whether to use the previous trained weights
    RUN = None # Which folder to find ckpt
    RESTORE_EPOCH = None # Which epoch to restore
    # Optimizer
    BATCH_NORM_DECAY = 0.9
    BATCH_NORM_EPSILON = 1e-5
    LEARNING_RATE = 3e-4
    BETA1 = 0.5

    # Training schedule
    PRE_TRAIN = False
    EPOCHS = None  # Num of epochs to train in the current run, override in sub-classes
    TRAIN_SIZE = None  # Num of samples used to train per epoch, override in sub-classes
    VAL_STEP = None

    SAVE_PER_EPOCH = 1  # How often to save the trained weights
    # Summary
    SUMMARY = True
    SUMMARY_GRAPH = True
    SUMMARY_SCALAR = True
    SUMMARY_IMAGE = False
    SUMMARY_HISTOGRAM = False

    # Results
    SAMPLE_DIR = None
    LOG_DIR = None
    WEIGHT_DIR = None

    # Debug
    DEBUG = False

    def __init__(self):
        """Set values of computed attributes."""
        self.MIN_QUEUE_EXAMPLES = self.BATCH_SIZE * 3
        self.IMAGE_DIM = [self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.CHANNEL]

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
            print("\n")

    def config_str(self):
        """Return a configurations string"""
        s = "\nConfigurations:\n"
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                s += "{:30} {}".format(a, getattr(self, a))
                s += "\n"
        return s


if __name__ == "__main__":
    tmp_config = Config()
    s = tmp_config.config_str()