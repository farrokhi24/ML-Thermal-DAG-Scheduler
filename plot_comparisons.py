import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_makespan_comparison(results_file="outputs/experiments/results.csv", output_dir="outputs/experiments/plots"):
    df = pd.read_csv(results_file)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    df_melted = pd.melt(df, 
                        id_vars=['cores', 'sample'], 
                        value_vars=['observed', 'he2019', 'zhao2020'],
                        var_name='Method', 
                        value_name='Makespan')
                        
    df_melted['Method'] = df_melted['Method'].map({
        'observed': 'Ours (Sub-Graph Division)',
        'he2019': 'He2019 (Longest Path)',
        'zhao2020': 'Zhao2020 (Critical Path First)'
    })

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='cores', y='Makespan', hue='Method', 
                capsize=.05, errorbar=('ci', 95))
    
    plt.title('Average Makespan vs Number of Cores')
    plt.xlabel('Number of Cores')
    plt.ylabel('Makespan (Simulation Time)')
    plt.legend(title='Scheduling Method')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(out_path / 'cores_vs_makespan.png', dpi=300)
    plt.close()
    
    print(f"Plot saved successfully in {out_path / 'cores_vs_makespan.png'}")

if __name__ == "__main__":
    plot_makespan_comparison()