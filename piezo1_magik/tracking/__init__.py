"""Pure-Python (NumPy) tracking post-processing shared by the scripts and the web interface; no PyTorch needed.

    linking      -- candidate graph building, greedy one-to-one linking of MAGIK edge probabilities,
                    track fragments and gap filling (fragment merging)
    track_stats  -- track lengths, temporal spans, gaps, fragmentation against ground truth, length categories
"""
