# import numpy as np
# import matplotlib.pyplot as plt

# # Data (prices of houses)
# x = [100, 200, 250, 300, 450, 100, 500]
# data = [8, 9, 11, 8.5, 6, 9.1, 1, 8.2, 10.1, 10, 3]

# # Calculate mean and standard deviation of the data
# m1 = np.mean(data)
# print(m1)
# std1 = np.std(data, ddof=0)
# samples = np.random.normal(m1,std1,1000)
# # Calculate the probability distribution (Gaussian PDF)
# prob_dis = []
# for i in range(len(samples)):
#     dis = (1 / (std1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((samples[i] - m1) / std1) ** 2)
#     prob_dis.append(dis)

# # Plot the result
# plt.plot(samples, prob_dis)
# plt.xlabel('Data')
# plt.ylabel('p(x)')
# plt.title('Gaussian Distribution')
# plt.show()

    
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Step 1: Generate Gaussian distributed data
# mean = 0  # Mean of the distribution
# std_dev = 1  # Standard deviation of the distribution
# data = np.random.normal(mean, std_dev, 1000)  # Generate 1000 points

# # Step 2: Plot the histogram of the data with a density curve
# sns.histplot(data)

# # Add labels and title
# plt.title('Gaussian Distribution - Bell Curve')
# plt.xlabel('Data')
# plt.ylabel('Density')

# # Show the plot
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Random data (replace with your actual data)
data = np.random.normal(loc=0, scale=1, size=1000)

# Step 1: Visualize the Data (Histogram and KDE)
sns.histplot(data, bins=30, kde=True)
plt.title('Histogram and KDE of Data')
plt.show()

# Step 2: Q-Q Plot for Normality
stats.probplot(data, dist="norm", plot=plt)
plt.title('Q-Q Plot for Normal Distribution')
plt.show()

# Step 3: Perform a statistical test (K-S test for normality)
stat, p_value = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
print(f"K-S Test Statistic: {stat}, p-value: {p_value}")

# Step 4: Anderson-Darling Test
result = stats.anderson(data, dist='norm')
print(f"Anderson-Darling Test Statistic: {result.statistic}")
print(f"Critical Values: {result.critical_values}")
