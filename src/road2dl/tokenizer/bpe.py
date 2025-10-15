import re
import json
from typing import Literal
import tiktoken
from collections import Counter, deque
from functools import lru_cache


def BPETokenizerTiktoken(gpt_version: Literal['gpt2', 'gpt4o'] = 'gpt2'):
    """
    Return a BPE tokenizer implemented by the tiktoken library, 
    which can specify the version used by gpt2 or gpt4o.
    
    Args:
        gpt_version (Literal['gpt2', 'gpt4o']): The GPT version to use.
    """
    match gpt_version:
        case 'gpt2':
            return tiktoken.get_encoding("gpt2")
        case 'gpt4o':
            return tiktoken.get_encoding("o200k_base")


class BPETokenizerSimple:
    """
    Simple BPE tokenizer implementation.
    """
    
    def __init__(self):
        # Maps token_id to token_str (e.g., {11246: "some"})
        self.vocab: dict[int, str] = {}
        # Maps token_str to token_id (e.g., {"some": 11246})
        self.inverse_vocab: dict[str, int] = {}
        # Dictionary of BPE merges: {(token_id1, token_id2): merged_token_id}
        self.bpe_merges: dict[tuple[int, int], int] = {}
        # For the official OpenAI GPT-2 merges, use a rank dict:
        # of form {(string_A, string_B): rank}, where lower rank = higher priority
        self.bpe_ranks: dict[tuple[str, str], int] = {}

    def train(self, text: str, vocab_size: int, allowed_special: set[str] = {"<|endoftext|>"}):
        """
        Train the BPE tokenizer on the given text from scratch.
        
        Args:
            text (str): The text to train the tokenizer on.
            vocab_size (int): The desired vocabulary size.
            allowed_special (set): A set of special tokens to allow in the vocabulary.
        """
        
        # Preprocess: Replace spaces with "Ġ"
        # Note that Ġ is a particularity of the GPT-2 BPE implementation
        # E.g., "Hello world" might be tokenized as ["Hello", "Ġworld"]
        # (GPT-4 BPE would tokenize it as ["Hello", " world"])
        processed_text = []
        for i, ch in enumerate(text):
            if ch == " " and i != 0:
                processed_text.append("Ġ")
            if ch != " ":
                processed_text.append(ch)
        processed_text = "".join(processed_text)
        
        # Initialize vocab with unique characters, including "Ġ" if present
        # Start with the first 256 ASCII characters
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(
            ch for ch in sorted(set(processed_text))
            if ch not in unique_chars
        )
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")
        self.vocab = {i: ch for i, ch in enumerate(unique_chars)}
        self.inverse_vocab = {ch: i for i, ch in self.vocab.items()}
        
        # Add allowed special tokens
        for token in allowed_special:
            if token not in self.inverse_vocab:
                new_id = len(self.vocab)
                self.vocab[new_id] = token
                self.inverse_vocab[token] = new_id
                
        # Tokenize the processed_text into token IDs
        token_ids = [self.inverse_vocab[ch] for ch in processed_text]
        
        # BPE steps 1-3: Repeatedly find and replace frequent pairs
        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None:
                break
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id
            
        # Build the vocabulary with merged tokens
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id
            
    @staticmethod
    def find_freq_pair(token_ids: list[int], mode="most"):
        pairs = Counter(zip(token_ids[:-1], token_ids[1:]))
        if not pairs:
            return None
        match mode:
            case "most":
                # return max(pairs.items(), key=lambda x: x[1])[0]
                return pairs.most_common(1)[0][0]
            case "least":
                # return min(pairs.items(), key=lambda x: x[1])[0]
                return pairs.most_common()[-1][0]
            case _:
                raise ValueError("Invalid mode. Choose 'most' or 'least'")
    
    @staticmethod
    def replace_pair(token_ids: list[int], pair_id: tuple[int, int], new_id: int):
        dq = deque(token_ids)
        replaced: list[int] = []
        
        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                dq.popleft()
            else:
                replaced.append(current)
        
        return replaced

    def load_vocab_and_merges_from_gpt2(self, vocab_path, bpe_merges_path):
        """
        Load pre-trained vocabulary and BPE merges from OpenAI's GPT-2 files.
        
        Args:
            vocab_path (str): Path to the vocab file (GPT-2 calls it 'encoder.json' or 'vocab.json').
            bpe_merges_path (str): Path to the bpe_merges file  (GPT-2 calls it 'vocab.bpe' or 'merges.txt').
        """
        
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            # Convert loaded vocabulary to correct format
            self.vocab = {int(v): k for k, v in loaded_vocab.items()}
            self.inverse_vocab = {k: int(v) for k, v in loaded_vocab.items()}

        # Handle newline character without adding a new token
        if "\n" not in self.inverse_vocab:
            # Use an existing token ID as a placeholder for '\n'
            # Preferentially use "<|endoftext|>" if available
            fallback_token = next(
                (token for token in ["<|endoftext|>", "Ġ", ""] if token in self.inverse_vocab), None
            )
            if fallback_token is not None:
                newline_token_id = self.inverse_vocab[fallback_token]
            else:
                # If no fallback token is available, raise an error
                raise KeyError("No suitable token found in vocabulary to map '\\n'.")
            self.inverse_vocab["\n"] = newline_token_id
            self.vocab[newline_token_id] = "\n"
            
        # Load GPT-2 merges and store them with an assigned "rank"
        self.bpe_ranks = {}  # reset ranks
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            if lines and lines[0].startswith("#"):
                lines = lines[1:]
            rank = 0
            for line in lines:
                pair = tuple(line.strip().split())
                if len(pair) == 2:
                    token1, token2 = pair
                    # If token1 or token2 not in vocab, skip
                    if token1 in self.inverse_vocab and token2 in self.inverse_vocab:
                        self.bpe_ranks[(token1, token2)] = rank
                        rank += 1
                    else:
                        print(f"Skipping pair {pair} as one token is not in the vocabulary.")

    def encode(self, text: str, allowed_special: set[str] | None = None):
        """
        Encode the input text into a list of token IDs, with tiktoken-style handling of special tokens.
    
        Args:
            text (str): The input text to encode.
            allowed_special (set or None): Special tokens to allow passthrough. If None, special handling is disabled.
    
        Returns:
            List of token IDs.
        """

        token_ids: list[int] = []
        
        # If special token handling is enabled
        if allowed_special is not None and len(allowed_special) > 0:
            # Build regex to match allowed special tokens
            special_pattern = (
                "(" + "|".join(re.escape(tok) for tok in sorted(allowed_special, key=len, reverse=True)) + ")"
            )
            # Encoding text while handling special tokens
            last_index = 0
            for mch in re.finditer(special_pattern, text):
                prefix = text[last_index : mch.start()]
                # Encode prefix without special handling
                token_ids.extend(self.encode(prefix, allowed_special=None))  
                # Encode special token
                special_token = mch.group(0)
                if special_token in self.inverse_vocab:
                    token_ids.append(self.inverse_vocab[special_token])
                else:
                    raise ValueError(f"Special token {special_token} not found in vocabulary.")
                last_index = mch.end()
            # Remaining part to process normally
            text = text[last_index:]  
            
        # If no special tokens, or remaining text after special token split:
        tokens: list[str] = []
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if i > 0:
                tokens.append("\n")
            words = line.split()
            for j, word in enumerate(words):
                if j == 0 and i > 0:
                    tokens.append("Ġ" + word)
                elif j == 0:
                    tokens.append(word)
                else:
                    tokens.append("Ġ" + word)

        for token in tokens:
            if token in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[token])
            else:
                token_ids.extend(self.tokenize_with_bpe(token))
    
        return token_ids
        
    def tokenize_with_bpe(self, token: str):
        """
        Tokenize a single token which is not in vocabulary using BPE merges.

        Args:
            token (str): The token to tokenize.

        Returns:
            list[int]: The list of token IDs after applying BPE.
        """
        
        # Tokenize the token into individual characters (as initial token IDs)
        raw_token_ids = [self.inverse_vocab.get(ch, None) for ch in token]
        token_ids = [tid for tid in raw_token_ids if tid is not None]
        if None in raw_token_ids:
            missing_chars = [ch for ch, tid in zip(token, raw_token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")
            
        # If we haven't loaded OpenAI's GPT-2 merges, use my approach
        if not self.bpe_ranks:
            can_merge = True
            while can_merge and len(token_ids) > 1:
                can_merge = False
                new_tokens: list[int] = []
                i = 0
                while i < len(token_ids) - 1:
                    pair = (token_ids[i], token_ids[i + 1])
                    if pair in self.bpe_merges:
                        merged_token_id = self.bpe_merges[pair]
                        new_tokens.append(merged_token_id)
                        # Merged pair {pair} -> {merged_token_id} ('{self.vocab[merged_token_id]}')
                        i += 2  # Skip the next token as it's merged
                        can_merge = True
                    else:
                        new_tokens.append(token_ids[i])
                        i += 1
                if i < len(token_ids):
                    new_tokens.append(token_ids[i])
                token_ids = new_tokens
            return token_ids
            
        # Otherwise, do GPT-2-style merging with the ranks:
        # 1) Convert token_ids back to string "symbols" for each ID
        symbols = [self.vocab[id_num] for id_num in token_ids]

        # Repeatedly merge all occurrences of the lowest-rank pair
        while True:
            # Collect all adjacent pairs
            pairs = set(zip(symbols[:-1], symbols[1:]))
            if not pairs:
                break

            # Find the pair with the lowest rank
            bigram = min(pairs, key=lambda x: self.bpe_ranks.get(x, float('inf')))

            # If no valid ranked pair is present, we're done
            if bigram is None or bigram not in self.bpe_ranks:
                break
            # else merge all occurrences of that pair
            first, second = bigram
            new_symbols: list[str] = []
            i = 0
            while i < len(symbols):
                # If we see (first, second) at position i, merge them
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i + 1] == second:
                    new_symbols.append(first + second)  # merged symbol
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
            if len(symbols) == 1:
                break

        # Finally, convert merged symbols back to IDs
        merged_ids = [self.inverse_vocab[sym] for sym in symbols]
        return merged_ids

    def decode(self, token_ids):
        """
        Decode a list of token IDs back into a string.

        Args:
            token_ids (list[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        buffer: list[str] = []
        for i, token_id in enumerate(token_ids):
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab.")
            token = self.vocab[token_id]
            if token == "\n":
                if buffer and not buffer[-1].endswith(" "):
                    buffer.append(" \n") # Add space if not present before a newline
            elif token.startswith("Ġ"):
                buffer.append(" " + token[1:])
            else:
                buffer.append(token)
        return "".join(buffer)
        
    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Save the vocabulary and BPE merges to JSON files.

        Args:
            vocab_path (str): Path to save the vocabulary.
            bpe_merges_path (str): Path to save the BPE merges.
        """
        # Save vocabulary
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2)

        # Save BPE merges as a list of dictionaries
        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [{"pair": list(pair), "new_id": new_id}
                           for pair, new_id in self.bpe_merges.items()]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)
            
    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Load the vocabulary and BPE merges from JSON files.

        Args:
            vocab_path (str): Path to the vocabulary file.
            bpe_merges_path (str): Path to the BPE merges file.
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        # Load BPE merges
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge["new_id"]
                self.bpe_merges[pair] = new_id
                
    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        """
        Get the ID of a special token.

        Args:
            token (str): The special token.

        Returns:
            int or None: The ID of the token if it exists, otherwise None.
        """
        return self.inverse_vocab.get(token, None)
    