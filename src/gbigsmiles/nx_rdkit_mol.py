# SPDX-License-Identifier: GPL-3
# Copyright (c) 2022-2025: Ludwig Schneider
# See LICENSE for details


def mol_graph_to_rdkit_mol(mol_graph):
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError("RDKit is  an optional dependency, but to generate RDKit molecules it is required. Please install RDKit for example with `pip install rdkit`.") from exc

    def convert_bond_type(bond_attr):
        if bond_attr["aromatic"]:
            return Chem.BondType.AROMATIC
        if bond_attr["bond_type"] == 1:
            return Chem.BondType.SINGLE
        if bond_attr["bond_type"] == 2:
            return Chem.BondType.DOUBLE
        if bond_attr["bond_type"] == 3:
            return Chem.BondType.TRIPLE
        if bond_attr["bond_type"] == 4:
            return Chem.BondType.QUADRUPLE

    _DIR_MAP = {"/": Chem.BondDir.ENDUPRIGHT, "\\": Chem.BondDir.ENDDOWNRIGHT}
    _CHIRAL_MAP = {
        "@": Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        "@@": Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    }

    mol = Chem.RWMol()
    graph_idx_to_mol_idx = {}
    chiral_atoms = []
    for graph_idx, data in mol_graph.nodes(data=True):
        atom = Chem.Atom(data["atomic_num"])
        atom.SetIsAromatic(data["aromatic"])
        atom.SetFormalCharge(data["charge"])
        graph_idx_to_mol_idx[graph_idx] = mol.AddAtom(atom)
        if data.get("chiral", "") in _CHIRAL_MAP:
            chiral_atoms.append(graph_idx)

    for u, v, attr in mol_graph.edges(data=True):
        begin_idx = graph_idx_to_mol_idx[u]
        end_idx = graph_idx_to_mol_idx[v]
        mol.AddBond(begin_idx, end_idx, convert_bond_type(attr))
        bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
        if bond is None:
            continue
        bond_dir = attr.get("bond_dir", "")
        if bond_dir in _DIR_MAP:
            bond.SetBondDir(_DIR_MAP[bond_dir])

    for graph_idx in chiral_atoms:
        _set_chiral_tag(mol, mol_graph, graph_idx, graph_idx_to_mol_idx, _CHIRAL_MAP)

    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return mol


def _set_chiral_tag(mol, mol_graph, graph_idx, graph_idx_to_mol_idx, chiral_map):
    """@/@@ is defined relative to the atom's SMILES neighbour order, while RDKit reads a
    chiral tag relative to the order of the atom's bonds (an implicit H counting last).
    The bonds here were added in graph order, so translate: sort the bonds by their SMILES
    slot (edge attribute ``nbr_order``) and invert the tag when that permutation is odd."""
    from rdkit import Chem

    data = mol_graph.nodes[graph_idx]
    atom = mol.GetAtomWithIdx(graph_idx_to_mol_idx[graph_idx])
    mol_idx_to_graph_idx = {v: k for k, v in graph_idx_to_mol_idx.items()}
    slots = [
        mol_graph.get_edge_data(graph_idx, mol_idx_to_graph_idx[bond.GetOtherAtomIdx(atom.GetIdx())])
        .get("nbr_order", {})
        .get(data.get("origin_idx", graph_idx))
        for bond in atom.GetBonds()
    ]
    tag = chiral_map[data["chiral"]]
    if None in slots or len(slots) not in (3, 4):
        import warnings

        warnings.warn(f"Cannot resolve the neighbour order of chiral atom {graph_idx}; its chiral tag is set as written and may be inverted.", stacklevel=3)
    else:
        if len(slots) == 3:
            slots.append(1)  # implicit H (or lone pair): SMILES slot 1, last in RDKit's order
        ranks = sorted(slots)
        perm = [ranks.index(s) for s in slots]
        if _permutation_is_odd(perm):
            tag = Chem.ChiralType.CHI_TETRAHEDRAL_CW if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW else Chem.ChiralType.CHI_TETRAHEDRAL_CCW
    atom.SetChiralTag(tag)


def _permutation_is_odd(perm) -> bool:
    odd, seen = False, set()
    for i in range(len(perm)):
        if i in seen:
            continue
        j, length = i, 0
        while j not in seen:
            seen.add(j)
            j = perm[j]
            length += 1
        odd ^= bool((length - 1) & 1)
    return odd
