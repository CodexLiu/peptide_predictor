import os
import subprocess
from pathlib import Path

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
        '--sampling_temp', '0.1 0.15 0.2',
        '--model_name', 'v_48_020',
        '--batch_size', '1',
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

def main():
    # Use the relaxed structure from your previous run
    relaxed_pdb = Path("design_output/cycle1/relaxed.pdb")
    
    if not relaxed_pdb.exists():
        print(f"Error: Could not find {relaxed_pdb}")
        return
    
    run_protein_mpnn(relaxed_pdb)

if __name__ == "__main__":
    main() 