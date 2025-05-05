# first line: 1
@memory.cache
def get_fingerprint_from_smiles(smiles):
    try:
        # PubChem → CID取得
        compounds = pcp.get_compounds(smiles, namespace="smiles")
        if compounds and compounds[0].cid:
            drug_data = pcp.Compound.from_cid(compounds[0].cid)
            if drug_data.fingerprint:
                bits = []
                for x in drug_data.fingerprint:
                    try:
                        bits.extend([int(b) for b in f"{int(x, 16):04b}"])
                    except Exception:
                        continue
                return np.array(bits, dtype=np.float32)
    except Exception as e:
        print(f"[PubChem FAIL] {smiles}: {e}")

    # フォールバック：ゼロで埋める（フェアなベースライン）
    return np.zeros(920, dtype=np.float32)
