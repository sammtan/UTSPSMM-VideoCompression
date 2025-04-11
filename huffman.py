import numpy as np
from collections import Counter

# Simple Huffman coding (Section 5.2.2)
def huffman_encode(data):
    # Build frequency table
    freq = Counter(data)
    if len(freq) <= 1:
        # Handle single-symbol case
        symbol = data[0] if len(data) > 0 else 0
        codebook = {symbol: "0"}
        encoded = "0" * len(data)
        return encoded, codebook
    
    # Build Huffman tree
    nodes = [(f, k) for k, f in freq.items()]
    while len(nodes) > 1:
        nodes.sort(key=lambda x: x[0])  # Sort by frequency
        f1, k1 = nodes.pop(0)
        f2, k2 = nodes.pop(0)
        nodes.append((f1 + f2, (k1, k2)))
    
    # Generate codes
    def build_codes(tree, prefix=""):
        if not isinstance(tree, tuple):  # Base case: single symbol
            return {tree: prefix}
        # Recursive case: unpack tuple
        k1, k2 = tree  # Tree is (symbol1, symbol2)
        codes = {}
        codes.update(build_codes(k1, prefix + "0"))
        codes.update(build_codes(k2, prefix + "1"))
        return codes
    
    codebook = build_codes(nodes[0][1])
    encoded = "".join(codebook[x] for x in data)
    return encoded, codebook

def huffman_decode(encoded, codebook):
    # Reverse codebook
    decodebook = {v: k for k, v in codebook.items()}
    decoded = []
    current_code = ""
    for bit in encoded:
        current_code += bit
        if current_code in decodebook:
            decoded.append(decodebook[current_code])
            current_code = ""
    return np.array(decoded, dtype=np.int16)