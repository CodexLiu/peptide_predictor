import subprocess
import os
from pathlib import Path
import re
from Bio import SeqIO
import shutil
import json
import time
import numpy as np
from collections import defaultdict
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional

@dataclass
class ResidueMemory:
    appearances: List[Dict[str, float]] = field(default_factory=list)
    frequency: int = 0

class PositionMemorySystem:
    def __init__(self, sequence_length: int):
        self.memory = {pos: {} for pos in range(sequence_length)}
        self.sequence_length = sequence_length
    
    def update_memory(self, position: int, residue: str, probability: float, cycle: int):
        if residue not in self.memory[position]:
            self.memory[position][residue] = ResidueMemory()
        
        mem = self.memory[position][residue]
        mem.appearances.append({"cycle": cycle, "probability": probability})
        mem.frequency += 1
    
    def calculate_residue_score(self, position: int, residue: str, current_cycle: int) -> float:
        mem = self.memory[position].get(residue)
        if not mem:
            return 0.0
        
        appearances = mem.appearances
        if not appearances:
            return 0.0
        
        base_score = max(app["probability"] for app in appearances)
        latest_appearance = max(app["cycle"] for app in appearances)
        recency_multiplier = 1.0 / (current_cycle - latest_appearance + 1)
        frequency_multiplier = mem.frequency / current_cycle
        
        return base_score * recency_multiplier * frequency_multiplier
    
    def get_best_historical_residue(self, position: int, current_cycle: int) -> Optional[str]:
        scores = {
            residue: self.calculate_residue_score(position, residue, current_cycle)
            for residue in self.memory[position]
        }
        if not scores:
            return None
        
        best_residue = max(scores.items(), key=lambda x: x[1])
        return best_residue[0] if best_residue[1] > 0.30 else None

def create_poly_gly_sequence(length):
    """Create a poly-glycine sequence of specified length"""
    return "G" * length

def write_fasta_input(protein_sequence, peptide_sequence, output_file, dimer=False):
    """Write the protein and peptide sequences to a FASTA file"""
    if dimer:
        fasta_content = f"""
>protein|name=long-protein
{protein_sequence}
>protein|name=long-protein-2
{protein_sequence}
>protein|name=peptide
{peptide_sequence}
""".strip()
    else:
        fasta_content = f"""
>protein|name=long-protein
{protein_sequence}
>protein|name=peptide
{peptide_sequence}
""".strip()
    
    with open(output_file, 'w') as f:
        f.write(fasta_content)

def run_chai_inference(fasta_path, output_dir):
    """Run the Chai inference script"""
    from chai_lab.chai1 import run_inference
    import torch
    
    candidates = run_inference(
        fasta_file=fasta_path,
        output_dir=output_dir,
        num_trunk_recycles=3,
        num_diffn_timesteps=200,
        seed=42,
        device=torch.device("cuda:0"),
        use_esm_embeddings=True,
    )
    return candidates

def convert_cif_to_pdb(cif_file, pdb_file):
    """Convert CIF file to PDB format"""
    from Bio import PDB
    parser = PDB.MMCIFParser()
    structure = parser.get_structure('structure', str(cif_file))
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(str(pdb_file))

def run_relax(input_pdb, output_pdb):
    """Run Rosetta relax protocol"""
    cmd = ['python', 'relax.py', input_pdb, '-o', output_pdb]
    subprocess.run(cmd, check=True)

def run_protein_mpnn(input_pdb):
    """Run ProteinMPNN on a single structure"""
    # Convert to absolute path
    input_pdb = os.path.abspath(input_pdb)
    print(f"Running ProteinMPNN on {input_pdb}")
    
    cmd = [
        'python', 'protein_mpnn_run.py',
        '--pdb_path', str(input_pdb),
        '--out_folder', 'output_sequences',
        '--num_seq_per_target', '64',
        '--sampling_temp', '0.1 0.15 0.2',  # Fixed temperature format
        '--model_name', 'v_48_020',
        '--batch_size', '64',
        '--seed', '42',
        '--backbone_noise', '0.02',
        '--save_score', '1',
        '--use_soluble_model'
    ]
    
    # Print the command for debugging
    print("Command:", ' '.join(cmd))
    
    # Change to ProteinMPNN directory
    original_dir = os.getcwd()
    os.chdir('ProteinMPNN')
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Output:", result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"Error running ProteinMPNN: {e}")
        print("Output:", e.stdout)
        print("Errors:", e.stderr)
    finally:
        os.chdir(original_dir)

def get_sequences_from_fa(fa_file):
    """Extract all sequences from the MPNN output file"""
    sequences = []
    try:
        with open(fa_file, 'r') as f:
            for line in f:
                if not line.startswith('>'):
                    # Extract the peptide part after the slash if it exists
                    sequence = line.strip()
                    if '/' in sequence:
                        peptide = sequence.split('/')[-1]
                    else:
                        peptide = sequence
                    sequences.append(peptide)
    except Exception as e:
        print(f"Warning: Error reading sequence file {fa_file}: {e}")
    return sequences

def create_position_frequency_matrix(sequences):
    """Calculate amino acid frequencies at each position"""
    seq_length = len(sequences[0])
    position_counts = defaultdict(lambda: defaultdict(int))
    
    for seq in sequences:
        for pos, aa in enumerate(seq):
            position_counts[pos][aa] += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(position_counts).fillna(0)
    
    # Convert counts to frequencies
    df = df / len(sequences)
    
    return df

def construct_consensus_peptide(freq_matrix, memory_system, cycle_num):
    """Construct a peptide using memory-aware consensus selection"""
    consensus_peptide = ""
    low_prob_positions = []
    
    # First pass: identify low probability positions
    for pos in range(freq_matrix.shape[1]):
        max_prob = freq_matrix[pos].max()
        if max_prob < 0.50:
            low_prob_positions.append(pos)
    
    # Second pass: construct peptide
    for pos in range(freq_matrix.shape[1]):
        max_prob = freq_matrix[pos].max()
        current_best_aa = freq_matrix[pos].idxmax()
        
        if max_prob >= 0.50:
            # High probability consensus - use it and update memory
            selected_aa = current_best_aa
            memory_system.update_memory(pos, selected_aa, max_prob, cycle_num)
        else:
            # Check historical memory
            historical_best = memory_system.get_best_historical_residue(pos, cycle_num)
            
            if historical_best:
                selected_aa = historical_best
            else:
                # Check if part of continuous low-prob region
                is_continuous = sum(1 for p in low_prob_positions if abs(p - pos) <= 1) >= 3
                selected_aa = 'G' if is_continuous else current_best_aa
        
        consensus_peptide += selected_aa
    
    return consensus_peptide

def run_design_cycle(cycle_num, protein_sequence, peptide_sequence, base_dir, memory_system, dimer=False):
    """Modified run_design_cycle that uses memory-aware peptide construction"""
    print(f"\nStarting design cycle {cycle_num}")
    
    # Create cycle directory
    cycle_dir = Path(base_dir) / f"cycle{cycle_num}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    
    # Write FASTA input
    fasta_path = cycle_dir / "input.fasta"
    write_fasta_input(protein_sequence, peptide_sequence, fasta_path, dimer=dimer)
    
    # Run Chai inference
    print("Running Chai inference...")
    output_dir = cycle_dir / "chai_output"
    candidates = run_chai_inference(fasta_path, output_dir)
    
    # Get best CIF file and convert to PDB
    cif_file = get_best_cif_file(output_dir)
    pdb_file = cycle_dir / "input.pdb"
    print(f"Converting CIF to PDB using {cif_file.name}...")
    convert_cif_to_pdb(cif_file, pdb_file)
    
    # Run Relax
    relaxed_pdb = cycle_dir / "relaxed.pdb"
    print("Running Rosetta relax...")
    run_relax(pdb_file, relaxed_pdb)
    
    # Run ProteinMPNN
    print("Running ProteinMPNN...")
    run_protein_mpnn(relaxed_pdb)
    
    # Get all sequences from MPNN output
    mpnn_fa_file = Path("ProteinMPNN/output_sequences/seqs") / f"{relaxed_pdb.stem}.fa"
    cycle_fa_file = cycle_dir / f"{relaxed_pdb.stem}.fa"
    shutil.copy2(mpnn_fa_file, cycle_fa_file)
    
    # Get all sequences and create frequency matrix
    sequences = get_sequences_from_fa(mpnn_fa_file)
    freq_matrix = create_position_frequency_matrix(sequences)
    
    # Save frequency matrix for analysis
    freq_matrix.to_csv(cycle_dir / "position_frequencies.csv")
    
    # Construct consensus peptide with memory system
    consensus_peptide = construct_consensus_peptide(freq_matrix, memory_system, cycle_num)
    
    print(f"Cycle {cycle_num} complete. Consensus sequence: {consensus_peptide}")
    return consensus_peptide

def get_best_cif_file(output_dir):
    """Get the CIF file with the best score from Chai output"""
    best_score = float('-inf')  # Higher score is better
    best_file = None
    
    # Look for all CIF files and their corresponding NPZ files
    output_dir = Path(output_dir)
    for cif_file in output_dir.glob('pred.model_idx_*.cif'):
        # Get corresponding NPZ file
        model_idx = cif_file.stem.split('_')[-1]
        npz_file = output_dir / f'scores.model_idx_{model_idx}.npz'
        
        try:
            if npz_file.exists():
                # Load and read the aggregate score
                with np.load(npz_file) as data:
                    score = float(data['aggregate_score'][0])
                    print(f"Model {model_idx} score: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_file = cif_file
        except Exception as e:
            print(f"Warning: Could not read score from {npz_file}: {e}")
            continue
    
    if best_file is None:
        raise FileNotFoundError("No valid CIF files with scores found in output directory")
    
    print(f"Selected model {best_file.stem} with aggregate score: {best_score:.4f}")
    return best_file

def get_final_model(protein_sequence, final_peptide, base_dir, dimer=False):
    """Generate the final structural model using the optimized peptide sequence"""
    print("\nGenerating final structural model...")
    
    # Create final model directory
    final_dir = Path(base_dir) / "final_model"
    final_dir.mkdir(parents=True, exist_ok=True)
    
    # Write final FASTA input
    fasta_path = final_dir / "final_input.fasta"
    write_fasta_input(protein_sequence, final_peptide, fasta_path, dimer=dimer)
    
    # Run Chai inference
    print("Running final Chai inference...")
    output_dir = final_dir / "chai_output"
    candidates = run_chai_inference(
        fasta_path=fasta_path,
        output_dir=output_dir
    )
    
    # Get best CIF file and convert to PDB
    best_cif = get_best_cif_file(output_dir)
    final_pdb = final_dir / "final_model.pdb"
    print("Converting final model to PDB format...")
    convert_cif_to_pdb(best_cif, final_pdb)
    
    # Run Relax on final model
    relaxed_final_pdb = final_dir / "final_model_relaxed.pdb"
    print("Running Rosetta relax on final model...")
    run_relax(final_pdb, relaxed_final_pdb)
    
    print(f"Final model generated and saved to: {relaxed_final_pdb}")
    return relaxed_final_pdb

def main():
    # Initial parameters
    num_cycles = 15
    peptide_length = 20
    base_dir = Path("design_output_memory_testing")
    dimer = True
    
    # Initialize memory system
    memory_system = PositionMemorySystem(peptide_length)
    
    # Protein sequence (your target protein sequence)
    protein_sequence = "PQITLWKRPLVTIKIGGQLKEALLDTGADDTVIEEMSLPGRWKPKMIGGIGGFIKVRQYDQIIIEIAGHKAIGTVLVGPTPVNIIGRNLLTQIGATLNF"
    
    # Initial poly-glycine peptide
    current_peptide = create_poly_gly_sequence(peptide_length)
    
    # Run design cycles with memory system
    for cycle in range(1, num_cycles + 1):
        current_peptide = run_design_cycle(
            cycle,
            protein_sequence,
            current_peptide,
            base_dir,
            memory_system,
            dimer=dimer
        )
    
    # Generate final model with the optimized peptide
    final_model_path = get_final_model(protein_sequence, current_peptide, base_dir, dimer=dimer)
    print(f"\nDesign pipeline complete!")
    print(f"Final peptide sequence: {current_peptide}")
    print(f"Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main() 