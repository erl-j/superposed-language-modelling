import torch

def first_occurrence(tensor, dim):
    """
    Takes a tensor of arbitrary shape and returns a tensor with the same shape up to
    the specified dimension (inclusive), marking first occurrences of unique slices
    along dimensions after dim with 1s, and 0s otherwise.
    
    Args:
        tensor (torch.Tensor): Input tensor of arbitrary shape
        dim (int): Dimension along which to find first occurrences
    
    Returns:
        torch.Tensor: Tensor with 1s for first occurrences and 0s otherwise
    """
    # Handle negative dim
    if dim < 0:
        dim = tensor.dim() + dim
    
    # Check valid dim
    if dim >= tensor.dim():
        raise ValueError(f"Dimension {dim} is out of bounds for tensor of shape {tensor.shape}")
    
    # If dim is the last dimension, every element is a first occurrence
    if dim == tensor.dim() - 1:
        return torch.ones(tensor.shape, dtype=torch.int, device=tensor.device)
    
    # Reshape tensor to combine dimensions before and after dim
    pre_shape = tensor.shape[:dim+1]
    post_shape = tensor.shape[dim+1:]
    
    # Reshape to (prod(pre_shape), prod(post_shape))
    reshaped = tensor.reshape(-1, torch.prod(torch.tensor(post_shape)).item())
    
    # Create output tensor initialized with zeros
    result = torch.zeros(reshaped.shape[0], dtype=torch.int, device=tensor.device)
    
    # Dictionary to track seen patterns
    seen_patterns = {}
    
    # For each pre-dim index
    for i in range(reshaped.shape[0]):
        # Get hashable representation of the post-dim slice
        # Convert to CPU to avoid device issues
        slice_bytes = reshaped[i].cpu().numpy().tobytes()
        
        # If this is the first occurrence
        if slice_bytes not in seen_patterns:
            seen_patterns[slice_bytes] = True
            result[i] = 1
    
    # Reshape result back to original dimensions up to dim
    return result.reshape(pre_shape)


def test_1d_tensor():
    """Test with a 1D tensor."""
    tensor = torch.tensor([1, 2, 2, 3, 1, 4])
    expected = torch.tensor([1, 1, 0, 1, 0, 1])
    result = first_occurrence(tensor, 0)
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"
    print("✓ 1D tensor test passed")


def test_2d_tensor_dim0():
    """Test with a 2D tensor, dim=0."""
    tensor = torch.tensor([
        [1, 2, 3],
        [1, 2, 3],
        [4, 5, 6],
        [1, 2, 3]
    ])
    expected = torch.tensor([1, 0, 1, 0])
    result = first_occurrence(tensor, 0)
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"
    print("✓ 2D tensor dim=0 test passed")


def test_2d_tensor_dim1():
    """Test with a 2D tensor, dim=1."""
    tensor = torch.tensor([
        [1, 1, 2, 3],
        [4, 4, 5, 6]
    ])
    expected = torch.tensor([
        [1, 0, 1, 1],
        [1, 0, 1, 1]
    ])
    result = first_occurrence(tensor, 1)
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"
    print("✓ 2D tensor dim=1 test passed")


def test_3d_tensor_dim0():
    """Test with a 3D tensor, dim=0."""
    tensor = torch.tensor([
        [[1, 2], [3, 4]],
        [[1, 2], [3, 4]],
        [[5, 6], [7, 8]]
    ])
    expected = torch.tensor([1, 0, 1])
    result = first_occurrence(tensor, 0)
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"
    print("✓ 3D tensor dim=0 test passed")


def test_3d_tensor_dim1():
    """Test with a 3D tensor, dim=1."""
    tensor = torch.tensor([
        [[1, 2], [1, 2], [3, 4]],
        [[5, 6], [5, 6], [7, 8]]
    ])
    expected = torch.tensor([
        [1, 0, 1],
        [1, 0, 1]
    ])
    result = first_occurrence(tensor, 1)
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"
    print("✓ 3D tensor dim=1 test passed")


def test_3d_tensor_dim2():
    """Test with a 3D tensor, dim=2."""
    tensor = torch.tensor([
        [[1, 1, 2], [3, 3, 4]],
        [[5, 5, 6], [7, 7, 8]]
    ])
    expected = torch.tensor([
        [[1, 0, 1], [1, 0, 1]],
        [[1, 0, 1], [1, 0, 1]]
    ])
    result = first_occurrence(tensor, 2)
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"
    print("✓ 3D tensor dim=2 test passed")


def test_complex_case():
    """Test with a more complex case."""
    # Create a tensor with repeating patterns
    tensor = torch.tensor([
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
        [[1, 2, 3], [4, 5, 6]],
        [[13, 14, 15], [16, 17, 18]]
    ])
    
    # Test dim=0 (first occurrence of entire 2D slices)
    expected_dim0 = torch.tensor([1, 1, 0, 1])
    result_dim0 = first_occurrence(tensor, 0)
    assert torch.equal(result_dim0, expected_dim0), f"Expected {expected_dim0}, got {result_dim0}"
    
    # Test dim=1 (first occurrence of rows within each batch)
    expected_dim1 = torch.tensor([
        [1, 1],
        [1, 1],
        [1, 1],
        [1, 1]
    ])
    result_dim1 = first_occurrence(tensor, 1)
    assert torch.equal(result_dim1, expected_dim1), f"Expected {expected_dim1}, got {result_dim1}"
    
    print("✓ Complex case test passed")


def test_edge_cases():
    """Test edge cases."""
    # Empty tensor
    tensor = torch.tensor([])
    try:
        result = first_occurrence(tensor, 0)
        assert False, "Should have raised an error for empty tensor"
    except ValueError:
        print("✓ Empty tensor test passed")
    
    # Scalar tensor
    tensor = torch.tensor(5)
    try:
        result = first_occurrence(tensor, 0)
        assert False, "Should have raised an error for scalar tensor"
    except ValueError:
        print("✓ Scalar tensor test passed")
    
    # Negative dim
    tensor = torch.tensor([[1, 2], [3, 4]])
    result = first_occurrence(tensor, -1)
    expected = torch.tensor([[1, 1], [1, 1]])
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"
    print("✓ Negative dim test passed")
    
    # Last dimension
    tensor = torch.tensor([[1, 2, 1], [3, 4, 3]])
    result = first_occurrence(tensor, 1)
    expected = torch.tensor([[1, 1, 0], [1, 1, 0]])
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"
    print("✓ Last dimension test passed")


def run_all_tests():
    """Run all test functions."""
    print("Running tests for first_occurrence function:")
    test_1d_tensor()
    test_2d_tensor_dim0()
    test_2d_tensor_dim1()
    test_3d_tensor_dim0()
    test_3d_tensor_dim1()
    test_3d_tensor_dim2()
    test_complex_case()
    test_edge_cases()
    print("All tests passed!")


if __name__ == "__main__":
    run_all_tests()