
# Earthquake Memory and Volatility Analysis (2010–2015, California Region)
# Supplementary Code for Continuity Theory Manuscript – Section 10

# 1. Load Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns

# 2. Load Data
# The dataset contains seismic events in California (magnitude ≥4.5) from 2010 to 2015.
df = pd.read_csv('2010-2015_ds.csv')
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')

# 3. Compute Inter-Event Times (in hours)
# Measures the time between consecutive earthquake events
df['inter_event_time_hrs'] = df['time'].diff().dt.total_seconds() / 3600
inter_event_times = df['inter_event_time_hrs'].dropna()

# 4. Plot 1: Histogram of Inter-Event Times vs. Poisson Model
# Validates deviation from memoryless (Poisson) rupture behavior
lambda_poisson = 1 / inter_event_times.mean()
x_vals = np.linspace(0, inter_event_times.max(), 200)
pdf_exp = lambda_poisson * np.exp(-lambda_poisson * x_vals)

plt.figure(figsize=(10, 6))
sns.histplot(inter_event_times, bins=40, stat='density', color='skyblue', label='Observed')
plt.plot(x_vals, pdf_exp, 'r--', label='Poisson (Exp) Model')
plt.title('Inter-Event Times (2010–2015) vs. Poisson Model')
plt.xlabel('Hours Between Earthquakes')
plt.ylabel('Density')
plt.legend()
plt.savefig('plot_2010_2015_histogram.png')
plt.close()

# 5. Plot 2: Magnitude Over Time
# Reveals episodic bursts and recursive rupture signatures
plt.figure(figsize=(10, 6))
plt.plot(df['time'], df['mag'], marker='o', linestyle='-', color='purple')
plt.title('Magnitude Over Time (2010–2015)')
plt.xlabel('Time')
plt.ylabel('Magnitude')
plt.grid(True)
plt.savefig('plot_2010_2015_magnitude.png')
plt.close()

# 6. Plot 3: Autocorrelation of Inter-Event Times (ACF)
# Statistically tests for memory (𝓥ₘ) presence in rupture sequences
plt.figure(figsize=(10, 6))
plot_acf(inter_event_times, lags=40, alpha=0.05)
plt.title('ACF of Inter-Event Times (2010–2015)')
plt.savefig('plot_2010_2015_acf.png')
plt.close()
