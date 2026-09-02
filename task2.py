import pandas as pd
import numpy as np
from scipy import stats

# Load dataset
df = pd.read_csv("team_stats_clean.csv")

print(df[["Team", "xG", "xGEff"]].head(10))
print("\nTotal teams:", len(df))
# Take a random sample of 36 teams
sample = df.sample(n=36, random_state=2029)

print("\nSample Size:")
print(len(sample))

print("\nSample Teams and xG Efficiency:")
print(sample[["Team", "xGEff"]])
# Get xG efficiency values
xg_eff = sample["xGEff"]

# Descriptive statistics
print("\nDescriptive Statistics:")
print(xg_eff.describe())

# Calculate 95% confidence interval
mean = xg_eff.mean()
std = xg_eff.std(ddof=1)
n = len(xg_eff)

standard_error = std / np.sqrt(n)
t_critical = stats.t.ppf(0.975, df=n-1)

lower_ci = mean - t_critical * standard_error
upper_ci = mean + t_critical * standard_error

print("\nMean xG Efficiency:", mean)
print("Standard Deviation:", std)
print("95% Confidence Interval:")
print("Lower CI:", lower_ci)
print("Upper CI:", upper_ci)
# Shapiro-Wilk normality test
shapiro_result = stats.shapiro(xg_eff)

print("\nShapiro-Wilk Normality Test:")
print("Statistic:", shapiro_result.statistic)
print("P-value:", shapiro_result.pvalue)

# One-sample t-test against xG efficiency = 1.00
t_test = stats.ttest_1samp(xg_eff, popmean=1.00)

print("\nOne-Sample T-Test:")
print("T-statistic:", t_test.statistic)
print("P-value:", t_test.pvalue)