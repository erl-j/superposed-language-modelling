from slm.util import top_p_probs
import torch

def test_topp():

    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    top_p = 1.0

    print(probs.shape)

    new_probs = top_p_probs(probs, top_p)

    assert torch.allclose(new_probs, torch.tensor([[0.1, 0.2, 0.3, 0.4]]))

    new_probs = top_p_probs(probs, 0.0)

    assert torch.allclose(new_probs, torch.tensor([[0.0, 0.0, 0.0, 1.0]]))
