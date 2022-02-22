import pytest
from ..cache import xor_bytes


@pytest.mark.parametrize(
    "b1,b2,r",
    [
        (b"\x12", b"\x34", b"\x26"),
        (b"\0\x12", b"\0\x34", b"\0\x26"),
        (b"\x12\0", b"\x34\0", b"\x26\0"),
        (b"\x12\xAB", b"\x34", b"\x26\xAB"),
        (b"\x12\xAB", b"\x34\xCD", b"\x26\x66"),
    ],
)
def test_xor_bytes(b1: bytes, b2: bytes, r: bytes) -> None:
    assert xor_bytes(b1, b2) == r
