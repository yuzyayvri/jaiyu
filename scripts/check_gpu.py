import torch

def main():
    print("🔍 Checking Jaiyu GPU/ROCm environment...")
    print(f"PyTorch version: {torch.__version__}")

    # ROCm uses the CUDA API in PyTorch, so torch.cuda.is_available() returns True
    is_available = torch.cuda.is_available()
    print(f"CUDA/ROCm available: {is_available}")

    if hasattr(torch.version, "hip"):
        print(f"HIP version: {torch.version.hip}")
    else:
        print("HIP version: Not found (Are you sure this is ROCm?)")

    if is_available:
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print("\nROCm is ready!")
    else:
        print("\nGPU not detected.")
        print("Check your ROCm installation, or make sure you exported:")
        print("export HSA_OVERRIDE_GFX_VERSION=10.3.0")

if __name__ == "__main__":
    main()
