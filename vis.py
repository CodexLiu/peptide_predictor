import os
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_peptide_sequences(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            if '/' in line:
                # Extract sequence after the '/'
                peptide = line.strip().split('/')[-1]
                sequences.append(peptide)
    return sequences

def create_position_frequency_matrix(sequences):
    # Find the length of peptides
    seq_length = len(sequences[0])
    
    # Create a dictionary to store counts for each position
    position_counts = defaultdict(lambda: defaultdict(int))
    
    # Count occurrences of each amino acid at each position
    for seq in sequences:
        for pos, aa in enumerate(seq):
            position_counts[pos][aa] += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(position_counts).fillna(0)
    
    # Convert counts to frequencies
    df = df / len(sequences)
    
    return df

def plot_heatmap(df, output_path):
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, cmap='YlOrRd', annot=True, fmt='.2f', cbar_kws={'label': 'Frequency'})
    plt.title('Amino Acid Frequency by Position')
    plt.xlabel('Position')
    plt.ylabel('Amino Acid')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # Create output directory for heatmaps
    os.makedirs('heatmaps', exist_ok=True)
    
    # Find all .fa files in design_output1 and its subdirectories
    for root, dirs, files in os.walk('design_output_failed2'):
        for file in files:
            if file.endswith('.fa'):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                
                # Extract sequences
                sequences = extract_peptide_sequences(file_path)
                
                if sequences:
                    # Create frequency matrix
                    freq_matrix = create_position_frequency_matrix(sequences)
                    
                    # Generate heatmap
                    output_filename = f"heatmaps/{os.path.basename(root)}_{file.replace('.fa', '_heatmap.png')}"
                    plot_heatmap(freq_matrix, output_filename)
                    print(f"Generated heatmap: {output_filename}")

if __name__ == "__main__":
    main()