import torch


class ByteTextEncoder:
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length

    def encode(self, text: str) -> torch.Tensor:
        encoded = text.encode("utf-8", errors="ignore")[: self.max_length]
        padded = encoded + b"\x00" * max(0, self.max_length - len(encoded))
        return torch.tensor(list(padded), dtype=torch.long)
