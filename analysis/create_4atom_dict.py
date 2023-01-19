from graphein.protein.config import ProteinGraphConfig
from graphein.protein.edges.atomic import add_atomic_edges, add_bond_order, add_ring_status
from graphein.protein.edges.distance import *
from graphein.protein.graphs import construct_graph
import networkx as nx

params_to_change = {"granularity": "atom", "edge_construction_functions": [add_atomic_edges, add_bond_order, add_ring_status], "edge_fns":[add_peptide_bonds]}
config = ProteinGraphConfig(**params_to_change)
BOND_TYPES = ["peptide"]

#create graph from PDB
g = construct_graph(config=config, pdb_code="135l")

#create all nonbranching subgraph of peptide bonds with 3 edges
DICT4A = []
for node in nx.dfs_preorder_nodes(g):
    for node2 in nx.dfs_preorder_nodes(g):
        if node != node2:
            if nx.has_path(g, node, node2):
                if nx.shortest_path_length(g, node, node2) == 3:
                    DICT4A.append(nx.shortest_path(g, node, node2))

atom_info = {'CA': 'Cm', 'C': 'Cm', 'O': 'Om', 'OXT': 'Om', 'N': 'Nm', 'CB': 'C', 'CG': 'C', 'CE': 'C', 'CD': 'C', 'CG1': 'C', 'CG2': 'C', 'CD1': 'C', 'CD2': 'C', 'CE1': 'C', 'CE2': 'C', 'CE3': 'C', 'CZ': 'C', 'CZ2': 'C', 'CZ3': 'C', 'CH2': 'C', 'ND1': 'N', 'ND2': 'N', 'NE': 'N', 'NE1': 'N', 'NE2': 'N', 'NH1': 'N', 'NH2': 'N', 'NZ': 'N', 'OD1': 'O', 'OD2': 'O', 'OE1': 'O', 'OE2': 'O', 'OG': 'O', 'OG1': 'O', 'OH': 'O', 'SD': 'S', 'SG': 'S'}

#fill out other parts of the dictionary
final_dict = {}
for node_tuple in DICT4A:
    residue_name = g.nodes[node_tuple[0]]['residue_name']
    atom_names = tuple(g.nodes[node]['atom_type'] for node in node_tuple)
    #create nested dictionary of atom names and atom types from atom_info
    atom_types = tuple(atom_info[atom_name] for atom_name in atom_names)
    if residue_name in final_dict:
        #if atom names and atom types
        if atom_names in final_dict[residue_name]:
            continue
        else:
            final_dict[residue_name][atom_names] = atom_types
    else:
        #add new entry to dictionary
        final_dict[residue_name] = {atom_names: atom_types}

with open('DICT4A_test2.py', 'w') as f:
    f.write('DICT4A = ' + repr(final_dict))
