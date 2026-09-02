import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("team_stats_clean.csv")

print(df.head())
print("\nColumns:")
print(df.columns)
# Calculate shot accuracy for each team
df["ShotAccuracy"] = (df["OnTarget"] / df["Attempts"]) * 100

print("\nShot Accuracy:")
print(df[["Team", "Attempts", "OnTarget", "ShotAccuracy"]].head(10))
# Find the median possession
median_possession = df["Possession"].median()

print("\nMedian Possession:")
print(median_possession)

# Divide teams into higher and lower possession groups
df["PossGroup"] = np.where(
    df["Possession"] >= median_possession,
    "Higher",
    "Lower"
)

print("\nNumber of teams in each group:")
print(df["PossGroup"].value_counts())
# Take 18 teams randomly from each group
higher_sample = df[df["PossGroup"] == "Higher"].sample(
    n=18,
    random_state=2026
)

lower_sample = df[df["PossGroup"] == "Lower"].sample(
    n=18,
    random_state=2026
)

# Combine both groups
sample = pd.concat([higher_sample, lower_sample])

print("\nSample size:")
print(len(sample))

print("\nSample group counts:")
print(sample["PossGroup"].value_counts())

print("\nSample teams:")
print(sample[["Team", "Possession", "ShotAccuracy", "PossGroup"]])
# Separate shot accuracy for both groups
higher_accuracy = sample[
    sample["PossGroup"] == "Higher"
]["ShotAccuracy"]

lower_accuracy = sample[
    sample["PossGroup"] == "Lower"
]["ShotAccuracy"]

# Descriptive statistics
print("\nHigher Possession - Descriptive Statistics:")
print(higher_accuracy.describe())

print("\nLower Possession - Descriptive Statistics:")
print(lower_accuracy.describe())

print("\nMean Shot Accuracy:")
print("Higher Possession:", higher_accuracy.mean())
print("Lower Possession:", lower_accuracy.mean())

print("\nDifference in Means:")
print(higher_accuracy.mean() - lower_accuracy.mean())
# 95% confidence interval for difference in means

mean_difference = higher_accuracy.mean() - lower_accuracy.mean()

se = np.sqrt(
    higher_accuracy.var(ddof=1) / len(higher_accuracy)
    +
    lower_accuracy.var(ddof=1) / len(lower_accuracy)
)

# Welch-Satterthwaite degrees of freedom
v1 = higher_accuracy.var(ddof=1)
v2 = lower_accuracy.var(ddof=1)
n1 = len(higher_accuracy)
n2 = len(lower_accuracy)

df_welch = (
    (v1/n1 + v2/n2) ** 2
    /
    (
        ((v1/n1) ** 2) / (n1 - 1)
        +
        ((v2/n2) ** 2) / (n2 - 1)
    )
)

t_critical = stats.t.ppf(0.975, df_welch)

lower_ci = mean_difference - t_critical * se
upper_ci = mean_difference + t_critical * se

print("\n95% Confidence Interval:")
print("Mean Difference:", mean_difference)
print("Lower CI:", lower_ci)
print("Upper CI:", upper_ci)
print("Welch Degrees of Freedom:", df_welch)
# Assumption tests

# Shapiro-Wilk normality test
shapiro_higher = stats.shapiro(higher_accuracy)
shapiro_lower = stats.shapiro(lower_accuracy)

print("\nShapiro-Wilk Normality Test:")
print("Higher Possession:", shapiro_higher)
print("Lower Possession:", shapiro_lower)

# Levene's test for equality of variances
levene_result = stats.levene(higher_accuracy, lower_accuracy)

print("\nLevene's Test:")
print(levene_result)
# Welch two-sample t-test
t_test = stats.ttest_ind(
    higher_accuracy,
    lower_accuracy,
    equal_var=False
)

print("\nWelch Two-Sample T-Test:")
print("T-statistic:", t_test.statistic)
print("P-value:", t_test.pvalue)