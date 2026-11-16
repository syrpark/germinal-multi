"""The search stage must carry no prior on the designable positions.

design_semigreedy decides by argmax over `seq["input"] * alpha + bias`, where
`seq["input"]` is a one-hot of the proposed mutant. A bias above `alpha` at a
position therefore pins that position absolutely: the proposal reverts to wild
type before it is ever scored.

A bias large against `alpha` makes the whole semigreedy stage a silent
no-op, burning `search_steps` GPU iterations to re-evaluate one sequence.
"""


import numpy as np

def test_argmax_reverts_whenever_bias_exceeds_alpha():
    """The invariant the fix rests on, stated numerically."""
    alpha = 2.0
    n_aa = 20
    wt, mutant = 3, 7

    def hard_choice(bias_at_wt):
        one_hot = np.zeros(n_aa)
        one_hot[mutant] = 1.0
        bias = np.zeros(n_aa)
        bias[wt] = bias_at_wt
        return int((one_hot * alpha + bias).argmax())

    # recover_bias=5 against alpha=2.0: the mutation never survives.
    assert hard_choice(5.0) == wt
    # Zeroed, as the search stage now does: the proposal reaches the loss.
    assert hard_choice(0.0) == mutant
    # The boundary, for the record -- anything at or above alpha pins it.
    assert hard_choice(alpha + 0.1) == wt
