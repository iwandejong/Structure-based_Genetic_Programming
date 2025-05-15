import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set Seaborn theme
sns.set_theme(style="darkgrid")

# File path
file_path = "outputs.csv"

try:
    # Read the CSV file
    df = pd.read_csv(file_path, comment="#")

    # Prepare the structure label for nicer legends
    structure_labels = {0: 'Regular', 1: 'Structured'}
    df['structureLabel'] = df['structured'].map(structure_labels)

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.lineplot(
        data=df,
        x="generation",
        y="bestTree",
        hue="structureLabel",   # Separate by structure type (Regular/Structured)
        style="structureLabel", # Use different line styles for structure type
        ci="sd",                # Use standard deviation for error bands
        estimator="mean",       # Plot mean value per generation
        palette="viridis",        # Color palette
    )

    # Labels and title
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Mean Fitness (BACC)", fontsize=12)
    plt.ylim(0.625, 0.875)

    # Save and show
    plt.tight_layout()
    plt.savefig("best_average_fitness_plot_seaborn.png", dpi=300)
    plt.show()

except Exception as e:
    print(f"Error: {e}")
