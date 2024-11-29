import subprocess
import os
from pathlib import Path
import re
from Bio import SeqIO
import shutil
import json
import time
import numpy as np

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
    # Create the downloads directory if it doesn't exist
    download_dir = Path.home() / '.local/lib/python3.10/site-packages/downloads'
    download_dir.mkdir(parents=True, exist_ok=True)
    
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

def run_protein_mpnn(input_pdb):
    """Run ProteinMPNN on a single structure"""
    # Convert to absolute path
    input_pdb = os.path.abspath(input_pdb)
    print(f"Running ProteinMPNN on {input_pdb}")
    
    cmd = [
        'python', 'protein_mpnn_run.py',
        '--pdb_path', str(input_pdb),
        '--out_folder', 'output_sequences',
        '--num_seq_per_target', '16',
        '--sampling_temp', '0.1 0.15 0.2',  # Fixed temperature format
        '--model_name', 'v_48_020',
        '--batch_size', '16',
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

def get_best_sequence_from_fa(fa_file):
    """Extract the best scoring sequence from the MPNN output file"""
    best_score = float('inf')
    best_sequence = None
    
    # Wait a moment for the file to be written
    import time
    time.sleep(2)
    
    try:
        with open(fa_file, 'r') as f:
            current_sequence = None
            for line in f:
                if line.startswith('>'):
                    # Extract score using regex
                    match = re.search(r'score=([\d.-]+)', line)
                    if match:
                        score = float(match.group(1))
                        if score < best_score:
                            best_score = score
                            # Get the sequence from the next line
                            sequence = next(f).strip()
                            # Extract the peptide part after the slash if it exists
                            if '/' in sequence:
                                peptide = sequence.split('/')[-1]
                            else:
                                peptide = sequence
                            best_sequence = peptide
    except Exception as e:
        print(f"Warning: Error reading sequence file {fa_file}: {e}")
    
    if best_sequence is None:
        print(f"Warning: No valid sequences found in {fa_file}")
        
    return best_sequence

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

def run_design_cycle(cycle_num, protein_sequence, peptide_sequence, base_dir, dimer=False):
    """Run one complete design cycle"""
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
    
    # Run ProteinMPNN
    print("Running ProteinMPNN...")
    run_protein_mpnn(pdb_file)
    
    # Get best sequence and copy the .fa file to cycle directory
    mpnn_fa_file = Path("ProteinMPNN/output_sequences/seqs") / f"{pdb_file.stem}.fa"
    cycle_fa_file = cycle_dir / f"{pdb_file.stem}.fa"
    shutil.copy2(mpnn_fa_file, cycle_fa_file)
    best_sequence = get_best_sequence_from_fa(mpnn_fa_file)
    
    print(f"Cycle {cycle_num} complete. Best sequence: {best_sequence}")
    return best_sequence

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
    # Remove the extra parameters since they're handled internally
    candidates = run_chai_inference(
        fasta_path=fasta_path,
        output_dir=output_dir
    )
    
    # Get best CIF file and convert to PDB
    best_cif = get_best_cif_file(output_dir)
    final_pdb = final_dir / "final_model.pdb"
    print("Converting final model to PDB format...")
    convert_cif_to_pdb(best_cif, final_pdb)
    
    print(f"Final model generated and saved to: {final_pdb}")
    return final_pdb

def main():
    # Initial parameters
    num_cycles = 5
    peptide_length = 10
    base_dir = Path("design_output")
    dimer = True  # Set this to True to enable dimer mode
    
    # Protein sequence (your target protein sequence)
    protein_sequence = "PQITLWKRPLVTIKIGGQLKEALLDTGADDTVIEEMSLPGRWKPKMIGGIGGFIKVRQYDQIIIEIAGHKAIGTVLVGPTPVNIIGRNLLTQIGATLNF"
    
    # Initial poly-glycine peptide
    current_peptide = create_poly_gly_sequence(peptide_length)
    
    # Run design cycles
    for cycle in range(1, num_cycles + 1):
        current_peptide = run_design_cycle(
            cycle,
            protein_sequence,
            current_peptide,
            base_dir,
            dimer=dimer
        )
    
    # Generate final model with the optimized peptide
    final_model_path = get_final_model(protein_sequence, current_peptide, base_dir, dimer=dimer)
    print(f"\nDesign pipeline complete!")
    print(f"Final peptide sequence: {current_peptide}")
    print(f"Final model saved to: {final_model_path}")

if __name__ == "__main__":
    main() 