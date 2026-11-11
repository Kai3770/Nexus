"""
NEXUS-3D-EXTREME :: engine/crypto.py
====================================
Dependency-free authenticated stream cipher built from the Python standard
library (SHA-256 in counter mode for the key-stream, HMAC-SHA-256 for the
integrity tag, encrypt-then-MAC).

It protects two things:

* the LAN discovery beacon, so that only game clients can read or forge the
  lobby advertisements travelling over UDP broadcast, and
* the body of the local profile file (see :mod:`engine.secure_save`).

Honest scope note: both keys are compiled into the game, exactly as the
specification demands ("hard-coded internal salt key").  That stops casual
snooping and hand-editing by colleagues; it is not a defence against somebody
who reads this source file.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import struct

import numpy as np

NONCE_SIZE = 12
TAG_SIZE = 16
_BLOCK = 32
_COUNTER = struct.Struct("<Q")


class AuthenticationError(Exception):
    """Raised when a sealed blob fails its integrity check."""


def derive_key(secret: bytes, purpose: bytes) -> bytes:
    """Domain separated sub-key: SHA-256(purpose || 0x00 || secret)."""
    return hashlib.sha256(purpose + b"\x00" + secret).digest()


def keystream(key: bytes, nonce: bytes, length: int) -> bytes:
    """``length`` pseudo random bytes: SHA-256(key || nonce || counter) blocks."""
    if length <= 0:
        return b""
    blocks = (length + _BLOCK - 1) // _BLOCK
    prefix = hashlib.sha256(key + nonce)
    out = bytearray()
    for counter in range(blocks):
        h = prefix.copy()
        h.update(_COUNTER.pack(counter))
        out += h.digest()
    return bytes(out[:length])


def xor_bytes(data: bytes, pad: bytes) -> bytes:
    """Vectorised XOR of two equally long byte strings."""
    if not data:
        return b""
    a = np.frombuffer(data, dtype=np.uint8)
    b = np.frombuffer(pad, dtype=np.uint8, count=a.size)
    return np.bitwise_xor(a, b).tobytes()


def seal(secret: bytes, plaintext: bytes, nonce: bytes | None = None) -> bytes:
    """
    Encrypt and authenticate.  Layout: ``nonce (12) | ciphertext | tag (16)``.
    """
    if nonce is None:
        nonce = os.urandom(NONCE_SIZE)
    if len(nonce) != NONCE_SIZE:
        raise ValueError("nonce must be %d bytes" % NONCE_SIZE)
    enc_key = derive_key(secret, b"nexus-enc")
    mac_key = derive_key(secret, b"nexus-mac")
    ciphertext = xor_bytes(plaintext, keystream(enc_key, nonce, len(plaintext)))
    tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()[:TAG_SIZE]
    return nonce + ciphertext + tag


def unseal(secret: bytes, blob: bytes) -> bytes:
    """Verify and decrypt a blob produced by :func:`seal`."""
    if len(blob) < NONCE_SIZE + TAG_SIZE:
        raise AuthenticationError("sealed blob too short")
    nonce = blob[:NONCE_SIZE]
    ciphertext = blob[NONCE_SIZE:-TAG_SIZE]
    tag = blob[-TAG_SIZE:]
    enc_key = derive_key(secret, b"nexus-enc")
    mac_key = derive_key(secret, b"nexus-mac")
    expected = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()[:TAG_SIZE]
    if not hmac.compare_digest(tag, expected):
        raise AuthenticationError("integrity tag mismatch")
    return xor_bytes(ciphertext, keystream(enc_key, nonce, len(ciphertext)))


def salted_sha256_hex(data: str, salt: str) -> str:
    """Hex SHA-256 of ``data`` followed by ``salt`` (the save-file signature)."""
    return hashlib.sha256((data + salt).encode("utf-8")).hexdigest()
