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
        origin_u, origin_v = (mol_graph.nodes[i].get("origin_idx", i) for i in (u, v))
        if origin_u == str(attr.get("bond_dir_to")) != origin_v:
            u, v = v, u  # a / or \ is read towards the atom it was written before
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
    _assign_stereo(mol)
    return mol


def _assign_stereo(mol):
    """RDKit's stereo clean-up clears chiral tags from atoms that are not stereocenters, but
    among centers that depend on each other (a symmetric fused ring) it also drops real ones
    for some atom orders.  A dropped tag goes back when the atom is a center by topology and
    the CIP labeler, which is exact, still labels it; a chain too large for the labeler keeps
    every such tag."""
    from rdkit import Chem
    from rdkit.Chem import rdCIPLabeler

    unset = Chem.ChiralType.CHI_UNSPECIFIED
    tags = {a.GetIdx(): a.GetChiralTag() for a in mol.GetAtoms() if a.GetChiralTag() != unset}
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    dropped = {i for i in tags if mol.GetAtomWithIdx(i).GetChiralTag() == unset}
    if dropped:
        bare = Chem.AddHs(mol)
        for atom in bare.GetAtoms():
            atom.SetChiralTag(unset)
        possible = {e.centeredOn for e in Chem.FindPotentialStereo(bare) if e.type == Chem.StereoType.Atom_Tetrahedral}
        dropped = sorted(dropped & possible)
    if not dropped:
        return
    probe = Chem.Mol(mol)
    for i, tag in tags.items():
        probe.GetAtomWithIdx(i).SetChiralTag(tag)
    try:
        rdCIPLabeler.AssignCIPLabels(probe, dropped, None, 1_250_000)
        dropped = [i for i in dropped if probe.GetAtomWithIdx(i).HasProp("_CIPCode")]
    except Exception:  # too many nodes or iterations
        pass
    for i in dropped:
        mol.GetAtomWithIdx(i).SetChiralTag(tags[i])
    Chem.AssignStereochemistry(mol, cleanIt=False, force=True)


def _set_chiral_tag(mol, mol_graph, graph_idx, graph_idx_to_mol_idx, chiral_map):
    """@/@@ is defined relative to the atom's SMILES neighbour order, while RDKit reads a
    chiral tag relative to the order of the atom's bonds (an implicit H counting last).
    A lone pair is not read like an implicit H: see the three-neighbour case below.
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
    tag, invert = chiral_map[data["chiral"]], False
    if None in slots or len(slots) not in (3, 4):
        import warnings

        warnings.warn(f"Cannot resolve the neighbour order of chiral atom {graph_idx}; its chiral tag is set as written and may be inverted.", stacklevel=3)
    else:
        if len(slots) == 3:
            atom.UpdatePropertyCache(strict=False)
            if atom.GetTotalNumHs():
                slots.append(1)  # implicit H: SMILES slot 1, last in RDKit's order
            else:  # lone pair: RDKit reads it last, inverted once per ring-closure digit on the atom
                invert = sum(1 < slot < 2 for slot in slots) % 2 == 1
                slots.append(max(slots) + 1)
        ranks = sorted(slots)
        perm = [ranks.index(s) for s in slots]
        if _permutation_is_odd(perm) != invert:
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
