import torch

class DeviceManager:

    def __init__(self) -> None:
        return
    
    def define_device(self):
        """Define the device to be used by PyTorch"""

        torch_version = torch.__version__

        # Print the PyTorch version
        print(f"PyTorch version: {torch_version}", end=" -- ")

        if torch.backends.mps.is_available():
            print("using MPS device on MacOS")
            defined_device = torch.device("mps")
        else:
            defined_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"using {defined_device}")

        return defined_device