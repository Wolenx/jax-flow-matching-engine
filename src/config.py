import jax.random as jrd

# Neural Network Configuration
IN_SIZE = 2
OUT_SIZE = 2
WIDTH = 64
DEPTH = 3
KEY = jrd.key(2026)
SEED = 1324  # Change this to test other possible outcomes of the training
SHAPE = "ring"  # choose between : ring, square, double latice,personalized.
# For the personnalized setup it is usable if one has a specific data set they want to get the matching distribution (Refere to README.md)

# Model visualization Config
HEATMAP_RESOLUTION = 1000
VECTORMAP_RESOLUTION = 50


# Choose the shape among the ones available : circle, square, archispiral.
