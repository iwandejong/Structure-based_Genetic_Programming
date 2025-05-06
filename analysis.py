import pandas as pd
import matplotlib.pyplot as plt
import time
import matplotlib.patches as mpatches
import numpy as np

# File path
file_path = "outputs.csv"

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))  # Slightly larger figure for better legend visibility

# Action type mapping for legend
action_types = {0: 'Reproduction', 1: 'Crossover', 2: 'Mutation'}
action_colors = {0: 'red', 1: 'green', 2: '#FF8800'}

# Plotting in real-time
plt.ion()  # Enable interactive mode

gens = 500

while True:
  try:
      # Read the CSV file
      df = pd.read_csv(file_path, comment="#")
      
      # Normalize the bestTree values for each run
      normalized_df = df.copy()
      for run, group in df.groupby("run"):
          # Normalize to [0,1] range for each run
          min_val = group["bestTree"].min()
          max_val = group["bestTree"].max()
          print(f"Run {run}: min_val = {min_val}, max_val = {max_val}")
          if max_val != min_val:  # Avoid division by zero
              normalized_df.loc[group.index, "bestTree"] = (group["bestTree"] - min_val) / (max_val - min_val)
          else:
              normalized_df.loc[group.index, "bestTree"] = 0  # If all values are the same
      
      ax.clear()  # Clear previous plot
      
      # Group by 'run' and plot each run using normalized values
      for run, group in normalized_df.groupby("run"):
          # This maps each individual point's action to its color
          colors = group["action"].map(action_colors)
          
          # Plot each point individually with its specific color
          for i in range(len(group)):
              ax.plot(group["generation"].iloc[i], group["bestTree"].iloc[i],
                      marker="o", linestyle="", color=colors.iloc[i])
          
          # Plot the connecting line separately (without markers)
          ax.plot(group["generation"], group["bestTree"],
                  marker="", linestyle="-", label=f"Run {run+1}")
      
      # Labels and title with dark mode aesthetics
      ax.set_xlabel("Generation")
      ax.set_ylabel("Normalized Fitness")
      ax.set_title("GP Population Fitness over Multiple Runs (Normalized per Run)")
      
      # Set y-axis limits for normalized data
      ax.set_ylim([-0.1, 1.1])
      
      # Create two legends
      # First legend for the runs
      leg1 = ax.legend(title="Run", loc="lower left", frameon=True)
      
      # Second legend for action types
      action_patches = [mpatches.Patch(color=color, label=action_types[action])
                        for action, color in action_colors.items()]
      ax.add_artist(leg1)  # Add first legend
      # ax.legend(handles=action_patches, title="Action Type", loc="upper right", frameon=True)
      
      # Set tick colors
      ax.tick_params(axis='x')
      ax.tick_params(axis='y')
      
      plt.draw()
      plt.pause(1)  # Update every second
      # plt.show()
      
  except Exception as e:
      print(f"Error reading file: {e}")
      plt.pause(1)  # Continue trying after error