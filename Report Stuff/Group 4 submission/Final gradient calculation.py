import numpy as np

def calculate_weighted_average(gradients_with_errors):
    """
    Calculate the weighted average of gradients and the error of this average.

    Parameters:
    gradients_with_errors (list of tuples): Each tuple contains (gradient, error)

    Returns:
    tuple: Weighted average gradient and its error.
    """
    N = len(gradients_with_errors)
    weighted_sum = sum(g / (e ** 2) for g, e in gradients_with_errors)
    weights_sum = sum(1 / (e ** 2) for _, e in gradients_with_errors)

    average_gradient1 = weighted_sum / weights_sum
    error_average_gradient1 = (1 / weights_sum) ** 0.5

    variance_gradient1 = sum((g - average_gradient1) ** 2 for g, _ in gradients_with_errors) / N

    return average_gradient1, error_average_gradient1, variance_gradient1

def calculate_average_and_error(gradients_with_errors):
    """
    Calculate the simple average of gradients and the error of this average without weighting.

    Parameters:
    gradients_with_errors (list of tuples): Each tuple contains (gradient, error)

    Returns:
    tuple: Average gradient and its error.
    """
    N = len(gradients_with_errors)
    average_gradient2 = sum(g for g, _ in gradients_with_errors) / N
    error_average_gradient2 = (sum(e ** 2 for _, e in gradients_with_errors) / N) ** 0.5
    variance_gradient2 = sum((g - average_gradient2) ** 2 for g, _ in gradients_with_errors) / N


    return average_gradient2, error_average_gradient2, variance_gradient2

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

gradients_with_errors = [
    (2.4647699728964154, 0.15000637049188653),
    (0.38257619893767236, 0.02870903591953865),
    (0.8528761423607352, 0.0735042388024122),
    (5.817794846456752, 1.115909881447803),
    (4.649986019988165, 0.19884480263009083),
    (0.9329810625436882, 0.07831651543841181),
    (0.13799933770047262, 0.010019531087146438),
    # Add more (gradient, error) tuples here
]

# Calculate average gradient
average_gradient = np.mean(gradients)

# Calculate standard error of the mean (SEM) as a measure of the accuracy
# SEM is the standard deviation divided by the square root of the number of observations
std_error = np.std(gradients, ddof=1) / np.sqrt(len(gradients))


average_gradient1, error_average_gradient1, variance_gradient1 = calculate_weighted_average(gradients_with_errors)
print(f"Average Weighted Gradient: {average_gradient1}, Error of the Average: {error_average_gradient1}, Variance of the Gradients: {variance_gradient1}")

average_gradient2, error_average_gradient2, variance_gradient2 = calculate_average_and_error(gradients_with_errors)
print(f"Average Gradient: {average_gradient2}, Error of the Average: {error_average_gradient2}, Variance of the Gradients: {variance_gradient2}")

print(average_gradient)
print(std_error)
