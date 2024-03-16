import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('benchmark.csv')

# Clean the data: remove unnecessary columns and rows with missing benchmark results
df = df.drop(columns=['error_occurred', 'error_message'])
df = df.dropna(subset=['cpu_time'])

# Extract the benchmark name for plotting
df['benchmark'] = df['name'].str.split('/').str[-1]

# Set the figure size
plt.figure(figsize=(10, 6))

# Plot each benchmark's CPU time as a bar
plt.bar(df['benchmark'], df['cpu_time'], color='skyblue')

# Add title, labels
plt.title('CPU Time for Each Benchmark')
plt.xlabel('Benchmark Operation')
plt.ylabel('CPU Time (ns)')

# Show grid and the plot
plt.grid(axis='y')
plt.show()
