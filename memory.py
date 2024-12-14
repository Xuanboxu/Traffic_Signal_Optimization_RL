import torch

class Memory:
    def __init__(self, size_max, size_min):
        self._samples = []  # Store samples as a list of PyTorch tensors
        self._size_max = size_max
        self._size_min = size_min

    def add_sample(self, sample):
        """
        Add a sample to the memory.
        """
        self._samples.append(torch.tensor(sample))
        if self._size_now() > self._size_max:
            self._samples.pop(0)  # Remove the oldest sample if memory exceeds size_max

    def get_samples(self, n):
        """
        Retrieve `n` samples randomly from memory.
        """
        if self._size_now() < self._size_min:
            return torch.tensor([])  # Return an empty tensor if memory is too small

        n = min(n, self._size_now())  # Ensure n does not exceed the current size
        indices = torch.randperm(self._size_now())[:n]  # Randomly sample indices
        return torch.stack([self._samples[i] for i in indices])

    def _size_now(self):
        """
        Get the current size of memory.
        """
        return len(self._samples)
