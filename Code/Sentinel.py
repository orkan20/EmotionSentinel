# Ideally we want the code to be robust: it shouldn't just die for no reason and then give up

# we also need it to be able to resume off of the database without losing any data: all "thinking" should be stored
# outside the memory of this algo

# We need to have the concept of LOW RANK INTEGRATION which will be able to handle both the database plus injecting
# matrices into a matrix (Recursion)

# Loss
# Token Accuracy
# Cosine Similarity?
# Perplexity
# KL Divergence

# ELBo (Evidence Lower Bound)
# Reconstruction Loss
# KL Divergence
# Perplexity
# Latent Space Smoothness? tells whether similar emotions are clustering together in latent space (dimensional model)

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json
import matplotlib as pyplot
import random

# with open('data.json', 'r') as file:
#     data = json.load(file)
# print(data)

import matplotlib.pyplot as plt
import numpy as np

# Your points (x, y, z)
x = [1, 2, 3, 4, 5]
y = [2, 3, 1, 5, 4]
z = [5, 3, 4, 1, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x, y, z)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Points')
plt.show()