import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv('outputs.csv')

# Extract unique runs
runs = data['run'].unique()

# Prepare results dataframe for individual runs
run_results = pd.DataFrame(columns=[
    'run', 'bt_mean', 'bt_min', 'bt_max', 'bt_std',
    'pop_mean', 'pop_min', 'pop_max', 'pop_std'
])

# print types of columns
print(data.dtypes)

# ! NGP
for run in runs:
    run_data = data[(data['run'] == run) & (data['structured'] == 0)]
    
    # For bestTree (BACC fitness)
    bt_mean = run_data['bestTree'].mean()
    bt_min = run_data['bestTree'].min()  # Minimum (best) fitness in this run
    bt_max = run_data['bestTree'].max()
    bt_std = run_data['bestTree'].std()
    
    # For population (BACC fitness)
    pop_mean = run_data['populationFitness'].mean()
    pop_min = run_data['populationFitness'].min()
    pop_max = run_data['populationFitness'].max()
    pop_std = run_data['populationFitness'].std()
    
    # Add to results
    run_results = pd.concat([run_results, pd.DataFrame({
        'run': [run],
        'bt_mean': [bt_mean],
        'bt_min': [bt_min],
        'bt_max': [bt_max],
        'bt_std': [bt_std],
        'pop_mean': [pop_mean],
        'pop_min': [pop_min],
        'pop_max': [pop_max],
        'pop_std': [pop_std]
    })], ignore_index=True)

# Now calculate cross-run statistics (these are the true statistics across runs)
# For bestTree
overall_bt_mean = run_results['bt_mean'].mean()  # Mean of run means
overall_bt_min = run_results['bt_min'].min()     # Best fitness across all runs
overall_bt_max = run_results['bt_max'].max()     # Worst best fitness across all runs
overall_bt_std = run_results['bt_mean'].std()    # Standard deviation of the mean fitnesses

# For population
overall_pop_mean = run_results['pop_mean'].mean() # Mean of run means
overall_pop_min = run_results['pop_min'].min()    # Best population fitness across all runs
overall_pop_max = run_results['pop_max'].max()    # Worst population fitness across all runs
overall_pop_std = run_results['pop_mean'].std()   # Standard deviation of the mean fitnesses

# Add overall statistics row
overall_row = pd.DataFrame({
    'run': ['Overall'],
    'bt_mean': [overall_bt_mean],
    'bt_min': [overall_bt_min],
    'bt_max': [overall_bt_max],
    'bt_std': [overall_bt_std],
    'pop_mean': [overall_pop_mean],
    'pop_min': [overall_pop_min],
    'pop_max': [overall_pop_max],
    'pop_std': [overall_pop_std]
})

# Combine with results
final_results = pd.concat([run_results, overall_row], ignore_index=True)

# ! SBGP
transfer_run_results = pd.DataFrame(columns=[
    'run', 'bt_mean', 'bt_min', 'bt_max', 'bt_std',
    'pop_mean', 'pop_min', 'pop_max', 'pop_std'
])

# Calculate transfer learning statistics for each run
for run in runs:
    transfer_data = data[(data['run'] == run) & (data['structured'] == 1)]
    if not transfer_data.empty:
        # For bestTree (BACC fitness) in transfer learning
        tl_bt_mean = transfer_data['bestTree'].mean()
        tl_bt_min = transfer_data['bestTree'].min()
        tl_bt_max = transfer_data['bestTree'].max() 
        tl_bt_std = transfer_data['bestTree'].std()
        
        # For population (BACC fitness) in transfer learning
        tl_pop_mean = transfer_data['populationFitness'].mean()
        tl_pop_min = transfer_data['populationFitness'].min()
        tl_pop_max = transfer_data['populationFitness'].max()
        tl_pop_std = transfer_data['populationFitness'].std()
        
        # Add to transfer learning results
        transfer_run_results = pd.concat([transfer_run_results, pd.DataFrame({
            'run': [run],
            'bt_mean': [tl_bt_mean],
            'bt_min': [tl_bt_min],
            'bt_max': [tl_bt_max],
            'bt_std': [tl_bt_std],
            'pop_mean': [tl_pop_mean],
            'pop_min': [tl_pop_min],
            'pop_max': [tl_pop_max],
            'pop_std': [tl_pop_std]
        })], ignore_index=True)

# Calculate cross-run statistics for transfer learning
if not transfer_run_results.empty:
    # For bestTree
    tl_overall_bt_mean = transfer_run_results['bt_mean'].mean()
    tl_overall_bt_min = transfer_run_results['bt_min'].min()
    tl_overall_bt_max = transfer_run_results['bt_max'].max()
    tl_overall_bt_std = transfer_run_results['bt_mean'].std()
    
    # For population
    tl_overall_pop_mean = transfer_run_results['pop_mean'].mean()
    tl_overall_pop_min = transfer_run_results['pop_min'].min()
    tl_overall_pop_max = transfer_run_results['pop_max'].max()
    tl_overall_pop_std = transfer_run_results['pop_mean'].std()
    
    # Add overall statistics row
    tl_overall_row = pd.DataFrame({
        'run': ['Overall'],
        'bt_mean': [tl_overall_bt_mean],
        'bt_min': [tl_overall_bt_min],
        'bt_max': [tl_overall_bt_max],
        'bt_std': [tl_overall_bt_std],
        'pop_mean': [tl_overall_pop_mean],
        'pop_min': [tl_overall_pop_min],
        'pop_max': [tl_overall_pop_max],
        'pop_std': [tl_overall_pop_std]
    })
    
    # Combine with transfer learning results
    final_transfer_results = pd.concat([transfer_run_results, tl_overall_row], ignore_index=True)

# Print LaTeX formatted table for all data
print("\\begin{table}[H]")
print(" \\centering")
print("\\resizebox{\\textwidth}{!}{")
print("\\begin{tabular}{lrrrrrrrr}")
print("\\toprule")
print("& \\multicolumn{4}{l}{\\emph{Best Tree Statistics (BACC Fitness)}} & \\multicolumn{4}{l}{\\emph{Population Statistics (BACC Fitness)}} \\\\")
print(" Run & Mean & Min & Max & Std & Mean & Min & Max & Std \\\\")
print("\\midrule")

for _, row in final_results.iterrows():
    run = row['run']
    if run == 'Overall':
        print("\\hline")
        print("\\hline")
    
    print(f" {run} & {row['bt_mean']:.6f} & {row['bt_min']:.6f} & {row['bt_max']:.6f} & {row['bt_std']:.6f} & {row['pop_mean']:.6f} & {row['pop_min']:.6f} & {row['pop_max']:.6f} & {row['pop_std']:.6f} \\\\")

print("\\bottomrule")
print("\\end{tabular}}")
print("\\caption{NGP\: Best Tree Fitness and Population Fitness statistics per GP run. The Overall row shows the mean across runs, the best result achieved in any run, the worst best result in any run, and the standard deviation of means across runs.}")
print("\\label{resultsTable}")
print("\\end{table}")

# Print LaTeX formatted table for transfer learning (after gen 50)
if not transfer_run_results.empty:
    print("\n\\begin{table}[H]")
    print(" \\centering")
    print("\\resizebox{\\textwidth}{!}{")
    print("\\begin{tabular}{lrrrrrrrr}")
    print("\\toprule")
    print("& \\multicolumn{4}{l}{\\emph{Best Tree Statistics (BACC Fitness)}} & \\multicolumn{4}{l}{\\emph{Population Statistics (BACC Fitness)}} \\\\")
    print(" Run & Mean & Min & Max & Std & Mean & Min & Max & Std \\\\")
    print("\\midrule")
    
    for _, row in final_transfer_results.iterrows():
        run = row['run']
        if run == 'Overall':
            print("\\hline")
            print("\\hline")
        
        print(f" {run} & {row['bt_mean']:.6f} & {row['bt_min']:.6f} & {row['bt_max']:.6f} & {row['bt_std']:.6f} & {row['pop_mean']:.6f} & {row['pop_min']:.6f} & {row['pop_max']:.6f} & {row['pop_std']:.6f} \\\\")
    
    print("\\bottomrule")
    print("\\end{tabular}}")
    print("\\caption{SGBP\: Best Tree Fitness and Population Fitness statistics per SBGP run. The Overall row shows the mean across runs, the best result achieved in any run, the worst best result in any run, and the standard deviation of means across runs.}")
    print("\\label{transferLearningTable}")
    print("\\end{table}")

# Create a visualization of fitness over generations
plt.figure(figsize=(12, 6))

gen_stats = data[data['structured'] == 0] # Filter for NGP
# Group by run and generation, then calculate mean values
gen_stats = gen_stats.groupby(['run', 'generation']).agg({
    'bestTree': 'mean',
    'populationFitness': 'mean'
}).reset_index()


# Plot best tree fitness per generation for each run
for run in runs:
    run_data = gen_stats[gen_stats['run'] == run]
    plt.plot(run_data['generation'], run_data['bestTree'], alpha=0.5, label=f'Run {run} Best')

# Calculate mean best tree fitness across all runs for each generation
mean_per_gen = gen_stats.groupby('generation').agg({
    'bestTree': 'mean'
}).reset_index()

# Plot mean best tree fitness across all runs
plt.plot(mean_per_gen['generation'], mean_per_gen['bestTree'], 
         color='black', linewidth=2, label='Mean Best')

plt.title('Best Tree Fitness per Generation (NGP)')
plt.xlabel('Generation')
plt.ylabel('Best Tree Fitness (BACC)')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('best_tree_fitness_over_time.png', dpi=300, bbox_inches='tight')

# Do the same for SBGP
plt.figure(figsize=(12, 6))

# Group by run and generation, then calculate mean values
gen_stats = data[data['structured'] == 1] # Filter for SBGP
gen_stats = gen_stats.groupby(['run', 'generation']).agg({
    'bestTree': 'mean',
    'populationFitness': 'mean'
}).reset_index()

# Plot best tree fitness per generation for each run
for run in runs:
    run_data = gen_stats[gen_stats['run'] == run]
    plt.plot(run_data['generation'], run_data['bestTree'], alpha=0.5, label=f'Run {run} Best')

# Calculate mean best tree fitness across all runs for each generation
mean_per_gen = gen_stats.groupby('generation').agg({
    'bestTree': 'mean'
}).reset_index()


# Plot mean best tree fitness across all runs
plt.plot(mean_per_gen['generation'], mean_per_gen['bestTree'], 
         color='black', linewidth=2, label='Mean Best')

plt.title('Best Tree Fitness per Generation (SBGP)')
plt.xlabel('Generation')
plt.ylabel('Best Tree Fitness (BACC)')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('best_tree_fitness_over_time_2.png', dpi=300, bbox_inches='tight')