from typing import AbstractSet, Collection, Literal, Protocol


class Tokenizer(Protocol):
    def encode(
        self, 
        text: str, 
        *, 
        allowed_special: AbstractSet[str] | Literal['all'] = set(),
        disallowed_special: Collection[str] | Literal['all'] = "all"
    ) -> list[int]: ...
    
    def decode(self, tokens: list[int]) -> str: ...
    
    
class PreTrainedTokenizer:
    ...
