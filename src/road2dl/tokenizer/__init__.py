from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...
    
    def decode(self, tokens: list[int]) -> str: ...