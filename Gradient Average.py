import numpy as np

# Gradient values provided
gradients = [
    2.4647699728964154,
    0.38257619893767236,
    0.8528761423607352,
    5.817794846456752,
    4.649986019988165,
    0.9329810625436882,
    0.13799933770047262
]

# Calculate average gradient
average_gradient = np.mean(gradients)

# Calculate standard error of the mean (SEM) as a measure of the accuracy
# SEM is the standard deviation divided by the square root of the number of observations
std_error = np.std(gradients, ddof=1) / np.sqrt(len(gradients))

print(average_gradient)
print(std_error)
