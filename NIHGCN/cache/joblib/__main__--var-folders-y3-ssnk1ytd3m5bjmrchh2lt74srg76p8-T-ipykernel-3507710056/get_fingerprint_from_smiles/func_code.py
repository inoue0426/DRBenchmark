# first line: 1
@memory.cache
def get_fingerprint_from_smiles(smiles, cid=None):
    fingerprint = None

    # Step 1: Try PubChem using SMILES
    compounds = pcp.get_compounds(smiles, namespace="smiles")
    if compounds and compounds[0].fingerprint is not None:
        print('use SMILES')
        fingerprint = ''.join(f"{int(hex_char,16):04b}" for hex_char in compounds[0].fingerprint)

    # Step 2: Try PubChem using CID (if available)
    elif cid is not None:
        print('use CID')
        compound = pcp.Compound.from_cid(cid)
        if compound and compound.fingerprint is not None:
            fingerprint = ''.join(f"{int(hex_char,16):04b}" for hex_char in compound.fingerprint)

    # Step 3: Use RDKit (fallback)
    if fingerprint is None:
        print('use RDKit')
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES and no PubChem fingerprint available")

        rdkit_fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=881)
        fingerprint = rdkit_fp.ToBitString()

    return np.array([int(bit) for bit in fingerprint])
