"""Local cost used to pack evidence into the answer-model context.

The installed runtime has no tokenizer for the answer model. ``tiktoken`` is
not a dependency, and the Hugging Face tokenizers that ship with the dense
extra belong to the embedding model, not to the answer model.

This counter is therefore an explicit upper-bound estimate:

    ceil(utf-8 byte length / 2)

Korean syllables are 3 bytes and are often about one BPE token, so dividing by
2 overestimates and keeps the packed context inside the budget. It is not a
character count, and it is not the provider's billed token count. The method
id is stored on the run record.
"""

from __future__ import annotations

CONTEXT_TOKEN_COUNTER_ID = "utf8_bytes_ceil_div_2"


def count_context_tokens(text: str) -> int:
    """Upper-bound context cost. Empty text costs nothing."""
    if not text:
        return 0
    return (len(text.encode("utf-8")) + 1) // 2
