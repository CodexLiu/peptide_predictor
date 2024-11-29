from Bio import PDB

def convert_cif_to_pdb(cif_file, pdb_file):
    parser = PDB.MMCIFParser()
    structure = parser.get_structure('structure', cif_file)
    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(pdb_file)
print('hi')

# Convert the file
convert_cif_to_pdb('cycle4.cif', 'input_cycle4.pdb')