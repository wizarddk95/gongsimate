"""Incremental reader for the grounded JSON object.

The Responses API streams this object as ``response.output_text.delta``
snapshots. A value is used only once its JSON token has closed. Truncation
returns what is already closed. A finished token of the wrong shape raises.
"""

from __future__ import annotations

from dataclasses import dataclass

from disclosure_agent.agent.errors import StructuredOutputInvalid

_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}
_WS = " \t\r\n"
_BOUNDARY = set(",}]" + _WS)


@dataclass(frozen=True)
class PartialClaim:
    evidence_ids: tuple[str, ...] | None
    text: str
    ids_closed: bool
    text_closed: bool
    closed: bool


@dataclass(frozen=True)
class PartialGrounded:
    abstain: bool | None
    claims: tuple[PartialClaim, ...]


def parse_partial_grounded(snapshot: str) -> PartialGrounded:
    """Read as much of the grounded object as the snapshot has closed."""
    index = _skip_ws(snapshot, 0)
    if index >= len(snapshot):
        return PartialGrounded(abstain=None, claims=())
    if snapshot[index] != "{":
        raise StructuredOutputInvalid("structured output is not an object")
    index += 1
    abstain: bool | None = None
    claims: list[PartialClaim] = []
    while True:
        index = _skip_ws(snapshot, index)
        if index >= len(snapshot) or snapshot[index] == "}":
            break
        if snapshot[index] == ",":
            index += 1
            continue
        if snapshot[index] != '"':
            raise StructuredOutputInvalid("expected an object field")
        key, index, key_closed = _read_string(snapshot, index)
        if not key_closed:
            break
        index = _skip_ws(snapshot, index)
        if index >= len(snapshot) or snapshot[index] != ":":
            break
        index = _skip_ws(snapshot, index + 1)
        if index >= len(snapshot):
            break
        if key == "abstain":
            value, index, closed = _read_bool(snapshot, index)
            if not closed:
                break
            abstain = value
        elif key == "claims":
            if snapshot[index] != "[":
                raise StructuredOutputInvalid("claims is not a list")
            claims, index = _read_claims(snapshot, index)
            if claims and not claims[-1].closed:
                break
        elif key == "limitations":
            if snapshot[index] != "[":
                raise StructuredOutputInvalid("limitations is not a list")
            _values, index, closed = _read_string_array(snapshot, index)
            if not closed:
                break
        else:
            raise StructuredOutputInvalid(f"unexpected field {key}")
    return PartialGrounded(abstain=abstain, claims=tuple(claims))


def _read_claims(text: str, index: int) -> tuple[list[PartialClaim], int]:
    index += 1
    claims: list[PartialClaim] = []
    while True:
        index = _skip_ws(text, index)
        if index >= len(text):
            break
        if text[index] == "]":
            return claims, index + 1
        if text[index] == ",":
            index += 1
            continue
        if text[index] != "{":
            raise StructuredOutputInvalid("claim is not an object")
        claim, index = _read_claim(text, index)
        claims.append(claim)
        if not claim.closed:
            break
    return claims, index


def _read_claim(text: str, index: int) -> tuple[PartialClaim, int]:
    index += 1
    evidence_ids: tuple[str, ...] | None = None
    ids_closed = False
    body = ""
    text_closed = False
    closed = False
    while True:
        index = _skip_ws(text, index)
        if index >= len(text):
            break
        if text[index] == "}":
            index += 1
            closed = True
            break
        if text[index] == ",":
            index += 1
            continue
        if text[index] != '"':
            raise StructuredOutputInvalid("expected a claim field")
        key, index, key_closed = _read_string(text, index)
        if not key_closed:
            break
        index = _skip_ws(text, index)
        if index >= len(text) or text[index] != ":":
            break
        index = _skip_ws(text, index + 1)
        if index >= len(text):
            break
        if key == "evidence_ids":
            if text[index] != "[":
                raise StructuredOutputInvalid("evidence_ids is not a list")
            values, index, array_closed = _read_string_array(text, index)
            if not array_closed:
                break
            evidence_ids = tuple(values)
            ids_closed = True
        elif key == "text":
            if text[index] != '"':
                raise StructuredOutputInvalid("claim text is not a string")
            body, index, text_closed = _read_string(text, index)
            if not text_closed:
                break
        else:
            raise StructuredOutputInvalid(f"unexpected claim field {key}")
    return (
        PartialClaim(
            evidence_ids=evidence_ids,
            text=body,
            ids_closed=ids_closed,
            text_closed=text_closed,
            closed=closed,
        ),
        index,
    )


def _read_string_array(text: str, index: int) -> tuple[list[str], int, bool]:
    index += 1
    values: list[str] = []
    while True:
        index = _skip_ws(text, index)
        if index >= len(text):
            return values, index, False
        if text[index] == "]":
            return values, index + 1, True
        if text[index] == ",":
            index += 1
            continue
        if text[index] != '"':
            raise StructuredOutputInvalid("list item is not a string")
        value, index, closed = _read_string(text, index)
        if not closed:
            return values, index, False
        values.append(value)


def _read_bool(text: str, index: int) -> tuple[bool | None, int, bool]:
    index = _skip_ws(text, index)
    rest = text[index:]
    for word, value in (("true", True), ("false", False)):
        if rest.startswith(word):
            after = index + len(word)
            if after >= len(text) or text[after] in _BOUNDARY:
                return value, after, True
            raise StructuredOutputInvalid("expected a boolean")
        if word.startswith(rest):
            return None, index, False
    if rest == "":
        return None, index, False
    raise StructuredOutputInvalid("expected a boolean")


def _read_string(text: str, index: int) -> tuple[str, int, bool]:
    if index >= len(text) or text[index] != '"':
        raise StructuredOutputInvalid("expected a string")
    index += 1
    chars: list[str] = []
    while index < len(text):
        char = text[index]
        if char == '"':
            return "".join(chars), index + 1, True
        if char == "\\":
            if index + 1 >= len(text):
                return "".join(chars), index, False
            escaped = text[index + 1]
            if escaped == "u":
                if index + 6 > len(text):
                    return "".join(chars), index, False
                hexpart = text[index + 2 : index + 6]
                if any(item not in "0123456789abcdefABCDEF" for item in hexpart):
                    raise StructuredOutputInvalid("invalid unicode escape")
                chars.append(chr(int(hexpart, 16)))
                index += 6
                continue
            if escaped not in _ESCAPES:
                raise StructuredOutputInvalid("invalid escape")
            chars.append(_ESCAPES[escaped])
            index += 2
            continue
        chars.append(char)
        index += 1
    return "".join(chars), index, False


def _skip_ws(text: str, index: int) -> int:
    while index < len(text) and text[index] in _WS:
        index += 1
    return index
