import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import defaultdict

# Define all possible amino acids in a fixed order
ALL_AAS = list('ACDEFGHIKLMNPQRSTVWY')

def extract_peptide_sequences(file_path):
    sequences = []
    with open(file_path, 'r') as f:
        for line in f:
            if '/' in line:
                peptide = line.strip().split('/')[-1]
                sequences.append(peptide)
    return sequences

def create_standardized_frequency_matrix(sequences):
    seq_length = len(sequences[0])
    matrix = np.zeros((len(ALL_AAS), seq_length))
    
    # Fill matrix with frequencies
    for pos in range(seq_length):
        counts = defaultdict(int)
        total = len(sequences)
        for seq in sequences:
            counts[seq[pos]] += 1
        for i, aa in enumerate(ALL_AAS):
            matrix[i][pos] = counts[aa] / total
    
    return matrix

def interpolate_matrices(matrix1, matrix2, t):
    return matrix1 * (1 - t) + matrix2 * t

class AnimatedHeatmap:
    def __init__(self, matrices, cycle_nums):
        self.matrices = matrices
        self.cycle_nums = cycle_nums
        # Create figure with two subplots (logo on top, heatmap below)
        self.fig, (self.ax_logo, self.ax) = plt.subplots(2, 1, figsize=(15, 15), 
                                                        gridspec_kw={'height_ratios': [1, 2]})
        self.frames_between = 20
        
        # Initialize both plots
        self.im = self.ax.imshow(self.matrices[0], aspect='auto', 
                                cmap='YlOrRd', interpolation='nearest',
                                vmin=0, vmax=1)
        
        # Setup heatmap
        self.ax.set_yticks(range(len(ALL_AAS)))
        self.ax.set_yticklabels(ALL_AAS)
        self.ax.set_xlabel('Position in Peptide')
        self.ax.set_ylabel('Amino Acid')
        
        plt.tight_layout()
    
    def init_animation(self):
        return [self.im]
    
    def animate(self, frame):
        total_frames = (len(self.matrices) - 1) * self.frames_between
        cycle = int(frame / self.frames_between)
        t = (frame % self.frames_between) / self.frames_between
        
        if cycle < len(self.matrices) - 1:
            interpolated = interpolate_matrices(
                self.matrices[cycle],
                self.matrices[cycle + 1],
                t
            )
            self.im.set_array(interpolated)
            
            # Clear previous bars
            self.ax_logo.clear()
            
            # Redraw sequence logo
            current_bottom = np.zeros(interpolated.shape[1])
            for pos in range(interpolated.shape[1]):
                bottom = 0
                for aa_idx, aa in enumerate(ALL_AAS):
                    height = interpolated[aa_idx, pos]
                    if height > 0.1:  # Only show AAs with >10% frequency
                        bar = self.ax_logo.bar(pos, height, bottom=bottom, width=0.8,
                                             color=plt.cm.tab20(aa_idx/20))
                        # Add amino acid letter in the middle of each bar
                        if height > 0.15:  # Only add text if bar is tall enough
                            self.ax_logo.text(pos, bottom + height/2, aa,
                                            ha='center', va='center',
                                            color='black', fontweight='bold')
                        bottom += height
            
            # Reset logo plot settings
            self.ax_logo.set_xlim(-0.5, interpolated.shape[1]-0.5)
            self.ax_logo.set_ylim(0, 1)
            self.ax_logo.set_title('Amino Acid Frequencies by Position', fontsize=12)
            
            self.ax.set_title(f'Cycle {self.cycle_nums[cycle]} → {self.cycle_nums[cycle + 1]}')
        
        return [self.im]

def main():
    base_dir = 'design_output_failed3'
    
    # Get all .fa files
    cycles = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.fa'):
                # Get parent folder name
                parent_folder = os.path.basename(root)
                try:
                    # Extract number from folder name containing 'cycle'
                    if 'cycle' in parent_folder.lower():
                        cycle_num = int(''.join(filter(str.isdigit, parent_folder)))
                        cycles.append((cycle_num, os.path.join(root, file)))
                        print(f"Found cycle {cycle_num}: {os.path.join(root, file)}")
                except ValueError:
                    print(f"Found .fa file but couldn't extract cycle number from: {os.path.join(root, file)}")
    
    print(f"Found {len(cycles)} .fa files")
    
    # Sort by cycle number
    cycles.sort(key=lambda x: x[0])
    
    # Process all cycles
    matrices = []
    cycle_nums = []
    
    for cycle_num, file_path in cycles:
        print(f"Processing cycle {cycle_num} from {file_path}")  # Debug print
        sequences = extract_peptide_sequences(file_path)
        print(f"Found {len(sequences)} sequences")  # Debug print
        if sequences:
            matrix = create_standardized_frequency_matrix(sequences)
            matrices.append(matrix)
            cycle_nums.append(cycle_num)
    
    print(f"Processed {len(matrices)} matrices")  # Debug print
    
    if not matrices:
        print("Error: No matrices were created. Check if the input files exist and contain valid sequences.")
        return
    
    # Create animation
    anim = AnimatedHeatmap(matrices, cycle_nums)
    animation = FuncAnimation(
        anim.fig,
        anim.animate,
        init_func=anim.init_animation,
        frames=(len(matrices) - 1) * anim.frames_between,
        interval=250,
        blit=False
    )
    
    # Save with higher quality
    animation.save('new.gif', 
                  writer='pillow', 
                  fps=4,
                  dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
