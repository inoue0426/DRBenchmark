# first line: 1
@memory.cache
def get_fingerprint_from_smiles(smiles):
    try:
        compounds = pcp.get_compounds(smiles, namespace="smiles")
        if compounds and compounds[0].fingerprint:
            return np.array([int(f"{int(c, 16):04b}") for c in compounds[0].fingerprint])
    except pcp.BadRequestError:
        pass  # 無視してRDKitへフォールバック

    mol = Chem.MolFromSmiles(smiles)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=881)
        return np.array([int(b) for b in fp.ToBitString()])

    raise ValueError(f"Invalid SMILES: {smiles}")
