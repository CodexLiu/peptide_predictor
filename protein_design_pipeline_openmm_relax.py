import subprocess
import os
from pathlib import Path
import re
from Bio import SeqIO
import shutil
import mdtraj as md
import openmm as mm
from openmm import app, unit
import numpy as np

def create_poly_gly_sequence(length):
    """Create a poly-glycine sequence of specified length"""
    return "G" * length

def write_fasta_input(protein_sequence, peptide_sequence, output_file):
    """Write the protein and peptide sequences to a FASTA file"""
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

def setup_openmm_system(pdb):
    """Setup OpenMM system with Amber force field"""
    forcefield = app.ForceField('amber14-all.xml', 'amber14/tip3p.xml')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    
    # Add hydrogens appropriate for pH 7.4
    modeller.addHydrogens(forcefield, pH=7.4)
    
    # Add solvent
    modeller.addSolvent(forcefield, padding=1.0*unit.nanometers)
    
    # Create system
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0*unit.nanometers,
        constraints=app.HBonds
    )
    
    return system, modeller

def add_restraints(system, modeller, strength=5.0):
    """Add positional restraints to backbone atoms"""
    force = mm.CustomExternalForce("k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k", strength * unit.kilocalories_per_mole/unit.angstroms**2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    
    for atom in modeller.topology.atoms():
        if atom.name in ['CA', 'C', 'N']:  # Backbone atoms
            force.addParticle(atom.index, modeller.positions[atom.index])
    
    system.addForce(force)
    return system

def run_relax(input_pdb, output_pdb):
    """Run OpenMM minimization and short MD for structure relaxation"""
    # Load PDB
    pdb = app.PDBFile(input_pdb)
    
    # Setup system
    system, modeller = setup_openmm_system(pdb)
    system = add_restraints(system, modeller)
    
    # Setup integrator and simulation
    integrator = mm.LangevinMiddleIntegrator(
        300*unit.kelvin,
        1/unit.picosecond,
        0.002*unit.picoseconds
    )
    
    # Create simulation
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    
    # Minimize
    print('Minimizing...')
    simulation.minimizeEnergy(maxIterations=1000)
    
    # Short MD simulation to relax structure
    print('Running short MD for relaxation...')
    simulation.step(1000)  # 2 ps
    
    # Get final positions and save
    positions = simulation.context.getState(getPositions=True).getPositions()
    app.PDBFile.writeFile(
        simulation.topology,
        positions,
        open(output_pdb, 'w')
    )

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

def get_best_sequence_from_fa(fa_file):
    """Extract the best scoring sequence from the MPNN output file"""
    best_score = float('inf')
    best_sequence = None
    
    with open(fa_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                # Extract score using regex
                match = re.search(r'score=([\d.]+)', line)
                if match:
                    score = float(match.group(1))
                    if score < best_score:
                        best_score = score
                        # Get the sequence from the next line
                        sequence = next(f).strip()
                        # Extract the peptide part after the slash
                        peptide = sequence.split('/')[-1]
                        best_sequence = peptide
    
    return best_sequence

def get_best_cif_file(output_dir):
    """Get the CIF file with the best score from Chai output"""
    best_score = float('inf')
    best_file = None
    
    for file in output_dir.glob('pred.model_idx_*.cif'):
        # Extract score from filename or parse file if needed
        # Currently using the first file as an example
        return file  # For now, just return the first file
        
    if best_file is None:
        raise FileNotFoundError("No CIF files found in output directory")
    return best_file

def run_design_cycle(cycle_num, protein_sequence, peptide_sequence, base_dir):
    """Run one complete design cycle"""
    print(f"\nStarting design cycle {cycle_num}")
    
    # Create cycle directory
    cycle_dir = Path(base_dir) / f"cycle{cycle_num}"
    cycle_dir.mkdir(parents=True, exist_ok=True)
    
    # Write FASTA input
    fasta_path = cycle_dir / "input.fasta"
    write_fasta_input(protein_sequence, peptide_sequence, fasta_path)
    
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
    print("Running OpenMM minimization and short MD for structure relaxation...")
    run_relax(pdb_file, relaxed_pdb)
    
    # Run ProteinMPNN
    print("Running ProteinMPNN...")
    run_protein_mpnn(relaxed_pdb)
    
    # Get best sequence and copy the .fa file to cycle directory
    mpnn_fa_file = Path("ProteinMPNN/output_sequences/seqs") / f"{relaxed_pdb.stem}.fa"
    cycle_fa_file = cycle_dir / f"{relaxed_pdb.stem}.fa"
    shutil.copy2(mpnn_fa_file, cycle_fa_file)
    best_sequence = get_best_sequence_from_fa(mpnn_fa_file)
    
    print(f"Cycle {cycle_num} complete. Best sequence: {best_sequence}")
    return best_sequence

def main():
    # Initial parameters
    num_cycles = 5
    peptide_length = 6
    base_dir = Path("design_output")
    
    # Protein sequence (your target protein sequence)
    protein_sequence = "PQITLWKRPLVTIKIGGQLKEALLDTGADDTVIEEMSLPGRWKPKMIGGIGGFIKVRQYDQIIIEIAGHKAIGTVLVGPTPVNIIGRNLLTQIGATLNF"
    #
    #chya
    #CGVPAIQPVLSGLSRIVNGEEAVPGSWPWQVSLQDKTGFHFCGGSLINENWVVTAAHCGVTTSDVVVAGEFDQGSSSEKIQKLKIAKVFKNSKYNSLTINNDITLLKLSTAASFSQTVSAVCLPSASDDFAAGTTCVTTGWGLTRYTNANTPDRLQQASLPLLSNTNCKKYWGTKIKDAMICAGASGVSSCMGDSGGPLVCKKNGAWTLVGIVSWGSSTCSTSTPGVYARVTALVNWVQQTLAAN
    # Initial poly-glycine peptide
    current_peptide = create_poly_gly_sequence(peptide_length)
    
    # Run design cycles
    for cycle in range(1, num_cycles + 1):
        current_peptide = run_design_cycle(
            cycle,
            protein_sequence,
            current_peptide,
            base_dir
        )

if __name__ == "__main__":
    main() 