import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# File path
file_path = "outputs.csv"

# Create figure and axis
fig, ax = plt.subplots(figsize=(13, 8))  # Larger figure for better legend visibility

# Action type mapping for legend
action_types = {0: 'Reproduction', 1: 'Crossover', 2: 'Mutation'}
action_colors = {0: 'red', 1: 'green', 2: '#FF8800'}

# Structure type mapping for different line styles
structure_types = {0: 'Regular', 1: 'Structured'}
structure_styles = {0: '-', 1: '--'}  # Solid for regular, dashed for structured

# Plotting in real-time
# plt.ion()  # Enable interactive mode

# while True:
try:
    # Read the CSV file
    df = pd.read_csv(file_path, comment="#")
    
    # Normalize the populationFitness values for each distinct run-structure combination
    normalized_df = df.copy()
    # for (run, struct), group in df.groupby(["run", "structured"]):
    #     # Normalize to [0,1] range for each distinct run-structure combination
    #     min_val = group["populationFitness"].min()
    #     max_val = group["populationFitness"].max()
    #     if max_val != min_val:  # Avoid division by zero
    #         normalized_df.loc[group.index, "populationFitness"] = (group["populationFitness"] - min_val) / (max_val - min_val)
    #     else:
    #         normalized_df.loc[group.index, "populationFitness"] = 0  # If all values are the same
    
    # Define color mapping for runs
    run_colors = {}
    available_colors = plt.cm.tab10.colors  # Using tab10 colormap for distinct colors
    for i, run in enumerate(normalized_df['run'].unique()):
        run_colors[run] = available_colors[i % len(available_colors)]
    
    ax.clear()  # Clear previous plot
    
    # Group by 'run' and 'structured' to plot each variation
    for (run, struct), group in normalized_df.groupby(["run", "structured"]):
        # This maps each individual point's action to its color
        action_color_points = group["action"].map(action_colors)
        
        # Get line style based on structure type
        line_style = structure_styles[struct]
        
        # Use the same color for both structured and regular variations of the same run
        run_line_color = run_colors[run]
        
        # Plot each point individually with its specific action color
        for i in range(len(group)):
            ax.plot(group["generation"].iloc[i], group["populationFitness"].iloc[i],
                  marker="o", linestyle="", color=action_color_points.iloc[i])
        
        # Plot the connecting line with run color and structure line style
        ax.plot(group["generation"], group["populationFitness"],
              marker="", linestyle=line_style, color=run_line_color,
              label=f"Run {run+1} ({structure_types[struct]})")
    
    # Labels and title
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fitness (BACC)", fontsize=12)
    ax.set_title("GP Population Fitness over Multiple Runs", fontsize=14)
    
    # Set y-axis limits for normalized data
    ax.set_ylim([0.4, 0.9])
    
    # Create legends
    # First legend for the runs and structure types
    leg1 = ax.legend(title="Run Information", loc="upper left", 
                    bbox_to_anchor=(1.02, 1.0), frameon=True, handlelength=1.5)

    # Second legend for action types
    action_patches = [mpatches.Patch(color=color, label=action_types[action])
                      for action, color in action_colors.items()]
    leg2 = ax.legend(handles=action_patches, title="Action Type", 
                    loc="center left", bbox_to_anchor=(1.02, 0.28),
                    frameon=True)

    # Third legend for structure types
    structure_patches = [plt.Line2D([0], [0], color='black', linestyle=style, label=structure_types[struct])
                        for struct, style in structure_styles.items()]
    leg3 = ax.legend(handles=structure_patches, title="Structure Type",
                    loc="lower left", bbox_to_anchor=(1.02, 0.097), frameon=True)

    # Add all legends to the axes
    ax.add_artist(leg1)
    ax.add_artist(leg2)
    ax.add_artist(leg3)

    # Adjust layout to make room on the right
    plt.tight_layout(rect=[0, 0, 0.82, 1])  # Reduce right space to leave room for legends

    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # plt.draw()
    # plt.pause(1)  # Update every second
    plt.savefig("population_fitness_plot.png", dpi=300)
    plt.show()
    
except Exception as e:
    print(f"Error: {e}")
    plt.pause(1)  # Continue trying after error