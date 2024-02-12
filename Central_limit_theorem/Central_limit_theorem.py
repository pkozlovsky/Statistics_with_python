import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def generate_heights(n_samples=1000, mean_height=170, std_dev=10):
    """
    Generates simulated heights following a normal distribution.

    Args:
        n_samples: The number of height samples to generate.
        mean_height: The average height (in centimeters).
        std_dev: The standard deviation of heights (in centimeters).

    Returns:
        A NumPy array containing the simulated heights.
    """
    #heights = np.random.beta(a=2, b=5, size=n_samples) * 40 + 150 
    #heights = np.random.exponential(scale=10, size=n_samples)
    heights = np.random.normal(loc=mean_height, scale=std_dev, size=n_samples)
    return heights

# Generate the dataset
heights = generate_heights() 
#print(heights)

print("Average height:", np.mean(heights))
print("Standard deviation:", np.std(heights))


# ---  Sampling and Calculations ---
n_samples = 100        # Number of random samples to draw
sample_size = 30       # Size of each sample

# Create an empty DataFrame to store sample means
sample_means = pd.DataFrame(columns=['mean'])

# Repeat the sampling process
for i in range(n_samples):
    sample = np.random.choice(heights, size=sample_size)  # Randomly select a sample
    sample_mean = np.mean(sample)
    sample_means.loc[i] = [sample_mean]  # Store the mean in the DataFrame

# Display the first few samples
print(sample_means.head())
# Export to CSV
sample_means.to_csv("sample_means.csv", index=False)

print("Sample means data frame exported to 'sample_means.csv'.")

# --- Confidence Interval Calculation ---
# Confidence Interval Calculation for T-distribution
confidence_level = 0.95
alpha = 1 - confidence_level
degrees_of_freedom = len(sample_means) - 1  

sample_meanT = sample_means['mean'].mean()
sample_std_errorT = stats.sem(sample_means['mean'])  # Standard error of the mean
print("T student sample_mean:", sample_meanT)
print("T student sample_std_error:", sample_std_errorT)

# Use a t-distribution for the confidence interval
t_multiplier = stats.t.ppf(1 - alpha/2, df=degrees_of_freedom)  

margin_of_error = t_multiplier * sample_std_errorT
lower_bound = sample_mean - margin_of_error
upper_bound = sample_mean + margin_of_error

print(f"T student Confidence interval for the mean at {confidence_level*100:.0f}%:",
      f"[{lower_bound:.4f}, {upper_bound:.4f}] cm")

# Confidence Interval Calculation (Standard Normal Distribution)
confidence_level = 0.95
alpha = 1 - confidence_level

sample_mean = sample_means['mean'].mean()
sample_std_error = stats.sem(sample_means['mean'])

# Use standard normal distribution (z-scores)
z_multiplier = stats.norm.ppf(1 - alpha/2)  

margin_of_error = z_multiplier * sample_std_error
lower_bound = sample_mean - margin_of_error
upper_bound = sample_mean + margin_of_error

print(f"Standard Normal Confidence interval for the mean at {confidence_level*100:.0f}%:",
      f"[{lower_bound:.4f}, {upper_bound:.4f}] cm")
print("sample_mean:", sample_mean)
print("sample_std_error:", sample_std_error)

# --- Distribution Plots ---
bin_size = 15  # Adjust this value to change the bin size
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create two side-by-side subplots

# Original heights histogram
axes[0].hist(heights, bin_size)
axes[0].set_xlabel("Height (cm)")
axes[0].set_ylabel("Number of people")
axes[0].set_title("Distribution of Simulated Heights")

# Sample means histogram
axes[1].hist(sample_means['mean'], bin_size)
axes[1].set_xlabel("Sample mean")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Distribution of Sample Means")

fig.tight_layout()  # Adjust layout to prevent overlapping
plt.show()

