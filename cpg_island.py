def analyze_cpg_island(sequence: str) -> dict:
    """
    Compute GC content and OE ratio of CpG.

    Args:
        - sequence (str): input sequence

    Return:
        - dict: GC content and OE ratio
    """
    seq = sequence.upper()
    length = len(seq)

    if length == 0:
        return {"length": 0, "gc_content": 0.0, "oe_ratio": 0.0, "is_gci": False}

    count_c = seq.count("C")
    count_g = seq.count("G")
    count_cg = seq.count("CG")

    gc_content = 100.0 * (count_c + count_g) / length

    if count_c == 0 or count_g == 0:
        oe_ratio = 0.0
    else:
        oe_ratio = (count_cg * length) / (count_c * count_g)

    is_cgi = (length >= 200) and (gc_content > 50.0) and (oe_ratio >= 60)

    return {
        "length": length,
        "gc_content": round(gc_content, 2),
        "oe_ratio": oe_ratio,
        "is_gci": is_cgi,
    }


def analyze_cgi_for_diff_states(hidden_path: list, observed_sequence: str) -> dict:
    """
    Compute summary CpG island statistics (GC content and O/E ratio) for each state.

    Args:
        - hidden_path (list): A list of hidden states
        - observed_sequence (str): A list of bases corresponding to the hidden states.
    
    Returns:
        - dict: A dictionary where keys are the hidden states found in hidden_path \
            and values are the CpG island statistics corresponding to the hidden states.   
    """
    if not hidden_path or len(observed_sequence) == 0:
        raise ValueError("input is empty")
    elif len(hidden_path) != len(observed_sequence):
        raise ValueError("hidden_path and observed_sequence should be the same length.")

    state_seq_map = {}
    cur = hidden_path[0]
    start = 0
    for i in range(1, len(hidden_path)):
        state = hidden_path[i]
        if state != cur:
            if cur not in state_seq_map:
                state_seq_map[cur] = ""
            state_seq_map[cur] += observed_sequence[start:i]

            start = i
            cur = state
    if cur not in state_seq_map:
        state_seq_map[cur] = ""
    state_seq_map[cur] += observed_sequence[start : len(hidden_path)]

    state_result_map = {}
    for state, seq in state_seq_map.items():
        state_result_map[state] = analyze_cpg_island(seq)

    return state_result_map


def cal_block_size(hidden_path: list) -> dict:
    """
    Compute block size for each state.

    Args:
        - hidden_path (list): A list of hidden states
    
    Returns:
        - dict: A dictionary where keys are the hidden states found in hidden_path \
            and values are the lists of block size corresponding to the hidden states.   
    """
    if not hidden_path:
        raise ValueError("input is empty")

    state_blocksize_map = {}
    cur = hidden_path[0]
    length = 1
    for i in range(1, len(hidden_path)):
        state = hidden_path[i]
        if state == cur:
            length += 1
        else:
            if cur not in state_blocksize_map:
                state_blocksize_map[cur] = []
            state_blocksize_map[cur].append(length)

            length = 1
            cur = state
    if cur not in state_blocksize_map:
        state_blocksize_map[cur] = []
    state_blocksize_map[cur].append(length)

    return state_blocksize_map
