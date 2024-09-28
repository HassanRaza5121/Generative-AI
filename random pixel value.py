import numpy as np

# Pixel values (e.g., grayscale values ranging from 0 to 255)
pixel_values = np.arange(256)

# Example probability distribution for the pixel values (must sum to 1)
# For simplicity, let's assume a uniform distribution over all values
probabilities = np.full(256, 1/256)

# Randomly select a pixel value based on the probability distribution
random_pixel = np.random.choice(pixel_values, p=probabilities)
print("Randomly selected pixel value:", random_pixel)
