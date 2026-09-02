import pandas as pd
import numpy as np
from scipy import stats

# Load player passing dataset
df = pd.read_csv("player_pass_stats_clean.csv")

print(df.head(10))

print("\nColumns:")
print(df.columns)

print("\nTotal players:")
print(len(df))

print("\nPlayers by position:")
print(df["Pos"].value_counts())
# Keep only defenders and midfielders
eligible = df[df["Pos"].isin(["DF", "MF"])]

print("\nEligible players:")
print(len(eligible))

print("\nEligible players by position:")
print(eligible["Pos"].value_counts())
# Stratified random sample
defenders = eligible[eligible["Pos"] == "DF"].sample(
    n=22,
    random_state=2027
)

midfielders = eligible[eligible["Pos"] == "MF"].sample(
    n=14,
    random_state=2027
)

sample = pd.concat([defenders, midfielders])

print("\nSample size:")
print(len(sample))

print("\nSample by position:")
print(sample["Pos"].value_counts())

print("\nSample players:")
print(sample[["Player", "Pos", "Accuracy"]])
# Separate passing accuracy by position
defender_accuracy = sample[
    sample["Pos"] == "DF"
]["Accuracy"]

midfielder_accuracy = sample[
    sample["Pos"] == "MF"
]["Accuracy"]

# Descriptive statistics
print("\nDefenders - Descriptive Statistics:")
print(defender_accuracy.describe())

print("\nMidfielders - Descriptive Statistics:")
print(midfielder_accuracy.describe())

print("\nMean Passing Accuracy:")
print("Defenders:", defender_accuracy.mean())
print("Midfielders:", midfielder_accuracy.mean())

print("\nDifference in Means:")
print(defender_accuracy.mean() - midfielder_accuracy.mean())
# 95% confidence interval for difference in means

mean_difference = defender_accuracy.mean() - midfielder_accuracy.mean()

v1 = defender_accuracy.var(ddof=1)
v2 = midfielder_accuracy.var(ddof=1)
n1 = len(defender_accuracy)
n2 = len(midfielder_accuracy)

se = np.sqrt(v1 / n1 + v2 / n2)

df_welch = (
    (v1 / n1 + v2 / n2) ** 2
    /
    (
        ((v1 / n1) ** 2) / (n1 - 1)
        +
        ((v2 / n2) ** 2) / (n2 - 1)
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
# Shapiro-Wilk normality tests
shapiro_def = stats.shapiro(defender_accuracy)
shapiro_mid = stats.shapiro(midfielder_accuracy)

print("\nShapiro-Wilk Normality Tests:")
print("Defenders:", shapiro_def)
print("Midfielders:", shapiro_mid)

# Levene's test
levene_result = stats.levene(
    defender_accuracy,
    midfielder_accuracy
)

print("\nLevene's Test:")
print(levene_result)

# Welch two-sample t-test
t_test = stats.ttest_ind(
    defender_accuracy,
    midfielder_accuracy,
    equal_var=False
)

print("\nWelch Two-Sample T-Test:")
print("T-statistic:", t_test.statistic)
print("P-value:", t_test.pvalue)