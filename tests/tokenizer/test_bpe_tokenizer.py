import pytest
from pathlib import Path
from ezdl.tokenizer.bpe import BPETokenizerSimple, BPETokenizerTiktoken
from ezdl.data.the_verdict import TheVerdictDataset


@pytest.mark.parametrize('input_text, token_ids', [
    ("Jack embraced beauty through art and life.", [424, 256, 654, 531, 302, 311, 256, 296, 97, 465, 121, 595, 841, 116, 287, 466, 256, 326, 972, 46]),
    ("Jack embraced beauty through art and life.<|endoftext|>", [424, 256, 654, 531, 302, 311, 256, 296, 97, 465, 121, 595, 841, 116, 287, 466, 256, 326, 972, 46, 257]),
    ("Jack embraced bea 74uty through art and life.<|endoftext|>", [424, 256, 654, 531, 302, 311, 256, 296, 97, 256, 55, 52, 465, 121, 595, 841, 116, 287, 466, 256, 326, 972, 46, 257])
])
def test_bpe_tokenizer_simple_with_the_verdict_dataset(input_text, token_ids):
    tokenizer = BPETokenizerSimple()
    text = TheVerdictDataset.get_raw_text()
    tokenizer.train(text, 1000)
    assert len(tokenizer.vocab) == 1000, f"Expected vocabulary size of 1000, got {len(tokenizer.vocab)}"
    assert len(tokenizer.bpe_merges) == 742, f"Expected BPE merges of 742, got {len(tokenizer.bpe_merges)}"
    
    token_ids = tokenizer.encode(input_text, allowed_special={"<|endoftext|>"})
    assert token_ids == token_ids
    decoded_text = tokenizer.decode(token_ids)
    assert decoded_text == input_text
    
    
@pytest.mark.parametrize('input_text', [
    "Jack embraced beauty through art and life.",
    "Jack embraced beauty through art and life.<|endoftext|>",
    "Jack embraced bea 74uty through art and life.<|endoftext|>"
])
def test_bpe_tokenizer_gpt2_equality(input_text):
    tokenizer_1 = BPETokenizerSimple()
    gpt2_dir = Path(__file__).parents[2] / 'models/gpt2'
    tokenizer_1.load_vocab_and_merges_from_gpt2(
        gpt2_dir / 'vocab.json', gpt2_dir / 'merges.txt'
    )
    tokenizer_2 = BPETokenizerTiktoken()
    
    token_ids_1 = tokenizer_1.encode(input_text, allowed_special={"<|endoftext|>"})
    token_ids_2 = tokenizer_2.encode(input_text, allowed_special={"<|endoftext|>"})
    assert token_ids_1 == token_ids_2
    decoded_text_1 = tokenizer_1.decode(token_ids_1)
    decoded_text_2 = tokenizer_2.decode(token_ids_2)
    assert decoded_text_1 == decoded_text_2.replace('<|endoftext|>', ' \n')
    assert decoded_text_1 == input_text.replace('<|endoftext|>', ' \n')


@pytest.mark.parametrize('input_text', [
    "  Jack embraced beauty through art and life.  ",
])
@pytest.mark.xfail(reason="BPETOkenizerSimple removes whitespace at both ends of the string when encoding, while the tokenizer in tiktoken does not.")
def test_bpe_tokenizer_simple_with_spaces_at_both_ends(input_text):
    tokenizer = BPETokenizerSimple()
    gpt2_dir = Path(__file__).parents[2] / 'models/gpt2'
    tokenizer.load_vocab_and_merges_from_gpt2(
        gpt2_dir / 'vocab.json', gpt2_dir / 'merges.txt'
    )
    token_ids = tokenizer.encode(input_text, allowed_special={"<|endoftext|>"})
    decoded_text = tokenizer.decode(token_ids)
    assert input_text == decoded_text