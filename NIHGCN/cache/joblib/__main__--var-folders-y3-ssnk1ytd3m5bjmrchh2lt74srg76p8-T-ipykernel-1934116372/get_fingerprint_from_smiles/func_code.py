# first line: 1
@memory.cache
def get_fingerprint_from_smiles(smiles):
    """
    SMILES文字列からPubChemフィンガープリントを取得
    :param smiles: SMILES文字列
    :return: 881次元のフィンガープリント(numpy array)
    """
    compounds = pcp.get_compounds(smiles, namespace="smiles")
    if not compounds:
        raise ValueError("SMILESに該当する化合物が見つかりません")

    compound = compounds[0]
    fingerprint = ""
    for hex_char in compound.fingerprint:
        fingerprint += f"{int(hex_char, 16):04b}"

    return np.array([int(bit) for bit in fingerprint])
