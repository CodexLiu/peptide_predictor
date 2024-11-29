import os
from Bio import SeqIO
import numpy as np
from collections import Counter

def extract_sequences(file_path):
    """Extract sequences from fasta file, taking only part after last slash."""
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        # Split by '/' and take the last part
        seq_parts = str(record.seq).split('/')
        sequences.append(seq_parts[-1])
    return sequences

def get_consensus_sequence(sequences):
    """Get consensus sequence and its probability at each position."""
    if not sequences:
        return None, None
    
    # Convert sequences to array for easier manipulation
    seq_array = np.array([list(seq) for seq in sequences])
    seq_length = len(sequences[0])
    
    consensus = []
    probabilities = []
    
    # For each position
    for pos in range(seq_length):
        # Count amino acids at this position
        counts = Counter(seq_array[:, pos])
        total = sum(counts.values())
        
        # Get most common AA and its probability
        most_common_aa, count = counts.most_common(1)[0]
        probability = count / total
        
        consensus.append(most_common_aa)
        probabilities.append(probability)
    
    return ''.join(consensus), probabilities

def main():
    base_dir = 'design_output'
    
    # Get all .fa files
    cycles = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.fa'):
                parent_folder = os.path.basename(root)
                try:
                    if 'cycle' in parent_folder.lower():
                        cycle_num = int(''.join(filter(str.isdigit, parent_folder)))
                        cycles.append((cycle_num, os.path.join(root, file)))
                except ValueError:
                    continue
    
    # Sort by cycle number
    cycles.sort(key=lambda x: x[0])
    
    print("\nConsensus Sequence Evolution (Final Segment):")
    print("-" * 80)
    print(f"{'Cycle':<6} {'Consensus Sequence':<30} {'Position-wise Probabilities'}")
    print("-" * 80)
    
    # Process each cycle
    for cycle_num, file_path in cycles:
        sequences = extract_sequences(file_path)
        consensus, probs = get_consensus_sequence(sequences)
        
        # Format probabilities for printing
        prob_str = ' '.join([f'{p:.2f}' for p in probs])
        
        print(f"{cycle_num:<6} {consensus:<30} {prob_str}")
    
    print("-" * 80)

if __name__ == "__main__":
    main() 