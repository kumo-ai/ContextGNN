import matplotlib.pyplot as plt
import pandas as pd 
df = pd.read_csv('scripts/stats.csv', sep=" ")

def generate_performance_gain(df): 
    # Calculate % gain = ((hybrid_test_score - id_test_score) / id_test_score) * 100
    df['val_gain_on_id'] = ((df['hybrid_val_score'] - df['id_val_score']) / df['id_val_score'])
    df['val_gain_on_shallow'] = ((df['hybrid_val_score'] - df['shallow_val_score']) / df['shallow_val_score'])
    df = df.sort_values(by='val_seen_ratio')
    df['val_seen_ratio'] = df['val_seen_ratio'].round(2).astype(str)
    df['dataset_task'] = df['Dataset'] + '_' + df['task'] + '_' + df['val_seen_ratio']
    print(df['val_gain_on_id'])
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(df['dataset_task'], df['val_gain_on_id'], color='blue', label='HybridGNN Gain On IDGNN')

    plt.scatter(df['dataset_task'], df['val_gain_on_shallow'], color='red', label='HybridGNN Gain On ShallowRHS')


    # Customize x-axis labels to show both 'dataset_task' and 'val_seen_ratio'
    #xticks_labels = [f"{row['dataset_task']}_{row['val_seen_ratio']:.2f}" for _, row in df.iterrows()]

    # Set x-ticks to be val_seen_ratio and labels to be the custom labels with dataset-task and val_seen_ratio
    #plt.xticks(df['dataset_task'], xticks_labels, rotation=45, ha='right')
    plt.xticks(rotation=45, ha='right')

    #for x in df['val_seen_ratio']:
    #    plt.axvline(x=x, color='gray', linestyle='--', alpha=0.5)
    # Add labels and title
    plt.xlabel('Dataset_Task_{RHS seen in training}')
    plt.ylabel('Performance Gain')
    plt.yscale('symlog', linthresh=0.1)
    plt.title('Gain of Hybrid Val Score Over IDGNN/ShallowGNN Val Score vs Validation Seen ratio')

    # Optionally, add grid lines for better readability
    plt.grid(True)
    plt.subplots_adjust(bottom=0.4)

    # Show the plot
    plt.legend()
    plt.show()
    plt.savefig(f'distribution_1.png', format='png',
                dpi=300)
    
def generate_performance_dist(df): 
    # Calculate % gain = ((hybrid_test_score - id_test_score) / id_test_score) * 100
    df['sum_of_idgnn_shallow'] = df['id_val_score'] + df['shallow_val_score']
    df['dataset_task_layers'] = df['Dataset'] + '-' + df['task'] + '-' + str(df['num_layers']) + '_layers'
    df = df.sort_values(by='val_seen_ratio')

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(df['val_seen_ratio'], df['sum_of_idgnn_shallow'], color='blue', label='Sum of IDGNN + ShallowRHSGNN')

    plt.scatter(df['val_seen_ratio'], df['hybrid_val_score'], color='red', label='HybridGNN')
    plt.scatter(df['val_seen_ratio'], df['id_val_score'], color='yellow', label='IDGNN')
    plt.scatter(df['val_seen_ratio'], df['shallow_val_score'], color='green', label='ShallowRHSGNN')



    # Customize x-axis labels to show both 'dataset_task' and 'val_seen_ratio'
    #xticks_labels = [f"{row['dataset_task_layers']}\n{row['val_seen_ratio']:.3f}" for _, row in df.iterrows()]

    # Set x-ticks to be val_seen_ratio and labels to be the custom labels with dataset-task and val_seen_ratio
    #plt.xticks(df['val_seen_ratio'], xticks_labels, rotation=45, ha='right')

    for x in df['val_seen_ratio']:
        plt.axvline(x=x, color='gray', linestyle='--', alpha=0.5)

    # Add labels and title
    plt.xlabel('Validation Seen ratio')
    plt.ylabel('Performance')
    plt.title('Performance Comparison')

    # Optionally, add grid lines for better readability
    plt.grid(True)

    # Show the plot
    plt.legend()
    plt.show()
    plt.savefig(f'distribution_2.png', format='png',
                dpi=300)

#generate_performance_dist(df)
generate_performance_gain(df)