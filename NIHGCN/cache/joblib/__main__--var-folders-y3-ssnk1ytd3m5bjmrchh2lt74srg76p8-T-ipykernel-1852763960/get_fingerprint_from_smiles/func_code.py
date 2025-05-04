# first line: 1
@memory.cache
def get_fingerprint_from_smiles(smiles):
    try:
        compounds = pcp.get_compounds(smiles, namespace="smiles")
        if compounds and compounds[0].fingerprint:
            return np.array([int(f"{int(c, 16):04b}") for c in compounds[0].fingerprint])
    except pcp.BadRequestError:
        pass

    # RDKit fallback
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # 最後の手段：sanitize無効化でMolを強制生成
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol:
            try:
                Chem.SanitizeMol(mol)  # 無理なら無視
            except:
                pass

    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=881)
        return np.array([int(b) for b in fp.ToBitString()])

    raise ValueError(f"Invalid SMILES: {smiles}")
