#!/usr/bin/env python3

import sys
import argparse
from pyrosetta import *

def parse_args():
    parser = argparse.ArgumentParser(description='Relax a structure from PDB')
    parser.add_argument('input_pdb', help='Input PDB file')
    parser.add_argument('--output', '-o', default=None, help='Output PDB file (default: input_relaxed.pdb)')
    parser.add_argument('--cycles', '-c', type=int, default=5, help='Number of relaxation cycles (default: 5)')
    return parser.parse_args()

def setup_scorefxn():
    """Create and return customized score function with constraints enabled"""
    # Use ref2015_cst instead of standard ref2015
    scorefxn = pyrosetta.rosetta.core.scoring.ScoreFunctionFactory.create_score_function('ref2015_cst')
    return scorefxn

def relax_structure(pose, scorefxn, cycles):
    """Perform relaxation with gentle constraints to maintain structure"""
    # Create movemap allowing both chains to move
    movemap = pyrosetta.MoveMap()
    movemap.set_bb(True)
    movemap.set_chi(True)
    
    # Set up FastRelax with constraints
    relax = pyrosetta.rosetta.protocols.relax.FastRelax()
    relax.set_scorefxn(scorefxn)
    relax.set_movemap(movemap)
    
    # Add gentle coordinate constraints for both chains
    constraints = pyrosetta.rosetta.protocols.constraint_movers.AddConstraintsToCurrentConformationMover()
    constraints.apply(pose)  # Apply to all residues
    
    # Set moderate weight for coordinate constraints - enough to maintain structure
    # but allow reasonable optimization
    scorefxn.set_weight(pyrosetta.rosetta.core.scoring.ScoreType.coordinate_constraint, 1.0)
    
    # Tell FastRelax to maintain starting coordinates
    relax.constrain_relax_to_start_coords(True)
    
    # Apply relaxation
    relax.apply(pose)
    return pose

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize PyRosetta
    init()
    
    # Set output filename if not specified
    output_file = args.output or args.input_pdb.replace('.pdb', '_relaxed.pdb')
    
    try:
        # Load structure
        print(f"Loading structure from {args.input_pdb}")
        pose = pose_from_pdb(args.input_pdb)
        
        # Setup scoring function
        scorefxn = setup_scorefxn()
        
        # Perform relaxation
        print(f"Relaxing structure with {args.cycles} cycles...")
        initial_score = scorefxn(pose)
        relaxed_pose = relax_structure(pose, scorefxn, args.cycles)
        final_score = scorefxn(relaxed_pose)
        
        # Save relaxed structure
        relaxed_pose.dump_pdb(output_file)
        print(f"Relaxed structure saved to {output_file}")
        print(f"Initial score: {initial_score:.2f}")
        print(f"Final score: {final_score:.2f}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 