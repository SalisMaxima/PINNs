import torch

def test_pytorch_cuda():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available! Testing with a simple tensor operation.")

        # Set device to GPU
        device = torch.device("cuda")  # Use the first CUDA device available

        # Create tensors
        x = torch.rand(10, 10, device=device)  # Create a random tensor directly on the GPU
        y = torch.ones(10, 10, device=device)  # Create a tensor of ones on the GPU

        # Perform a simple addition
        z = x + y

        # Print the result
        print("Result of tensor addition:")
        print(z)
        print("Test completed successfully, CUDA is working with PyTorch.")
    else:
        print("CUDA is not available. Please check your installation.")

if __name__ == "__main__":
    test_pytorch_cuda()
