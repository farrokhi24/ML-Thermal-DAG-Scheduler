import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_thermal_trace(schedule_csv_path, output_csv_path, 
                           T_amb=45.0, R=0.8, C=3.0, dt=1.0, P_idle=2.0, seed=42):
    schedule_df = pd.read_csv(schedule_csv_path)
    
    np.random.seed(seed)
    unique_nodes = schedule_df['node'].unique()
    node_power = {node: np.round(np.random.uniform(10.0, 20.0), 2) for node in unique_nodes}
    schedule_df['power'] = schedule_df['node'].map(node_power)

    max_time = int(schedule_df['finish'].max())
    cores = sorted(schedule_df['core'].unique())
    time_steps = np.arange(0, max_time + 1, dt)
    
    temperatures = {core: [T_amb] for core in cores}
    powers = {core: [P_idle] for core in cores}
    
    alpha = np.exp(-dt / (R * C))
    for t in time_steps[1:]:
        for core in cores:
            active_task = schedule_df[(schedule_df['core'] == core) & 
                                      (schedule_df['start'] <= (t-1)) & 
                                      (schedule_df['finish'] > (t-1))]
            
            P_current = active_task.iloc[0]['power'] if not active_task.empty else P_idle
            
            T_prev = temperatures[core][-1]
            T_inf = P_current * R + T_amb
            T_new = T_prev * alpha + T_inf * (1 - alpha)
            
            temperatures[core].append(T_new)
            powers[core].append(P_current)

    out_df = pd.DataFrame({'Time': time_steps})
    for core in cores:
        out_df[f'Temp_Core_{core}'] = temperatures[core]
        out_df[f'Power_Core_{core}'] = powers[core]
    out_df.to_csv(output_csv_path, index=False)
    
    plt.figure(figsize=(12, 6))
    for core in cores:
        plt.plot(time_steps, temperatures[core], label=f'Core {core}', linewidth=2)
        
    plt.axhline(y=T_amb, color='gray', linestyle='--', label='Ambient Temp')
    plt.title('Core Temperatures over Time (RC Model)')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    generate_thermal_trace(
        schedule_csv_path='outputs/paper_demo/schedule.csv',
        output_csv_path='outputs/paper_demo/thermal_trace.csv'
    )
