# first line: 1
@memory.cache
def get_fingerprint_from_smiles(smiles):
    compounds = pcp.get_compounds(smiles, namespace="smiles")
    if not compounds:
        raise ValueError("No compounds related to the SMILES")

    compound = compounds[0]
    fingerprint = ""
    for hex_char in compound.fingerprint:
        fingerprint += f"{int(hex_char, 16):04b}"

    return np.array([int(bit) for bit in fingerprint])
