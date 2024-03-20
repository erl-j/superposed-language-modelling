import numpy as np
import torch
import torch.nn.functional as F

def loop_sm(sm, loop_bars, n_loops):
    '''
    4/4 only
    '''
    sm = sm.copy()
    # get resolution
    tpq = sm.ticks_per_quarter

    # get duration of a bar in ticks
    loop_duration = tpq * 4 * loop_bars
    
    for track in sm.tracks:
        new_notes = []
        for loop_idx in range(1,n_loops):
            for note in track.notes:
                note = note.copy()
                note.start = note.start + loop_duration*(loop_idx)
                new_notes.append(note)
        track.notes.extend(new_notes)
    return sm

def piano_roll(sm):
    sm = sm.copy()
    sm = sm.resample(tpq=4, min_dur=0)

    # set all is_drum to False
    for track in sm.tracks:
        track.is_drum = False

    pr = sm.pianoroll(modes=["frame"]).sum(axis=0).sum(axis=0)

    # flip y axis
    pr = pr[::-1]

    return pr

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float("Inf")):
    # from https://gist.github.com/bsantraigi/5752667525d88d375207f099bd78818b
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (vocabulary size)
        top_k >0: keep only top k tokens with highest probability (top-k filtering).
        top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)

    Basic outline taken from https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 2  # [BATCH_SIZE, VOCAB_SIZE]
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k, dim=1)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)

    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # Replace logits to be removed with -inf in the sorted_logits
    sorted_logits[sorted_indices_to_remove] = filter_value
    # Then reverse the sorting process by mapping back sorted_logits to their original position
    logits = torch.gather(sorted_logits, 1, sorted_indices.argsort(-1))

    return logits