"""
tans_encode.py — tANS (table ANS) checkpoint compressor
========================================================

Encodes all 8 DeepSpeed ZeRO-3 optimizer shards (weights, exp_avg, exp_avg_sq)
into real tANS bitstreams, using TWO strategies in parallel so we can compare:

  (A) PER-TENSOR  — each tensor gets its own probability table
  (B) CLUSTERED   — tensors are grouped by feature similarity; every tensor
                    in a cluster shares one table

Outputs land in:
    <CHECKPOINT_DIR>/tANS/
        per_tensor/
            rank<r>_<stream>.bin          # compressed bitstream
            rank<r>_<stream>_meta.pkl     # tables + shapes needed for decode
        clustered/
            rank<r>_<stream>.bin
            rank<r>_<stream>_meta.pkl
        report.txt                        # side-by-side comparison

tANS implementation
-------------------
We use a vectorised numpy tANS encode/decode that is exact (lossless):

  TABLE_LOG = 12  →  L = 4096 states, symbols encoded to log2(L/p_s) bits on avg

  Build:
    1. Quantise float probs → integer freqs summing exactly to L  (Knuth–Plass rounding)
    2. Spread symbols across the L slots (Yann Collet's "fast spread")
    3. Encode table: for each (state, symbol) → (new_state, n_bits, bits_out)
    4. Decode table: state → (symbol, new_state)

  Encode:  vectorised numpy loop over symbol stream; state transitions via
           pre-built encode table; bit output packed into uint64 words.

  Decode:  vectorised numpy loop reading bits back; state transitions via
           decode table; exact reconstruction.

Performance notes
-----------------
* ProcessPoolExecutor with spawn workers — each worker owns one tensor/rank shard
* All workers return only (compressed_bytes, metadata_dict) — no big arrays over IPC
  except the compressed bitstream itself (unavoidable)
* ThreadPoolExecutor used inside each worker for the encode loop chunks
* numpy vectorised inner loops → releases GIL → true thread parallelism

Tested correctness guarantee
----------------------------
After encode, decode is called immediately on the bitstream and the result is
compared element-by-element to the original uint32 array.  Any mismatch raises
a RuntimeError before writing any output.
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import hashlib
import multiprocessing as mp

# Must be set BEFORE any fork/spawn can happen
mp.set_start_method("spawn", force=True)

import numpy as np
import torch
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from sklearn.cluster import MiniBatchKMeans

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

CHECKPOINT_DIR = "/p/lustre5/kogiou1/llama-3-8b/8_GPUs/math/DS/epoch_0"
NUM_RANKS      = 8
TABLE_LOG      = 12          # tANS table size = 2**TABLE_LOG = 4096 states
N_WORKERS      = min(64, mp.cpu_count())   # process-pool workers
VERIFY_DECODE  = True        # roundtrip-check every encoded tensor

OUT_DIR        = os.path.join(CHECKPOINT_DIR, "tANS")

# ──────────────────────────────────────────────────────────────────────────────
# tANS TABLE CONSTRUCTION
# ──────────────────────────────────────────────────────────────────────────────

TABLE_LOG_MIN, TABLE_LOG_MAX = 8, 16
 
try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f): return f
        return deco
 
 
def choose_table_log(n_vocab: int) -> int:
    if n_vocab <= 1:
        return 4
    return max(TABLE_LOG_MIN, min(TABLE_LOG_MAX, int(np.ceil(np.log2(n_vocab)))))
 
 
def _quantise_freqs(counts: np.ndarray, L: int) -> np.ndarray:
    n = len(counts)
    total = int(counts.sum())
    raw   = counts.astype(np.float64) * (L / total)
    freqs = np.maximum(np.floor(raw).astype(np.int64), 1)
    frac  = raw - np.floor(raw)
    diff  = L - int(freqs.sum())
    if diff > 0:
        order = np.argsort(-frac)
        i = 0
        while diff > 0:
            freqs[order[i % n]] += 1
            diff -= 1; i += 1
    elif diff < 0:
        order = np.argsort(frac)
        need, i, guard = -diff, 0, 0
        while need > 0:
            j = order[i % n]
            if freqs[j] > 1:
                freqs[j] -= 1; need -= 1
            i += 1; guard += 1
            if guard > 50 * n + 10_000:
                raise RuntimeError("quantise_freqs: failed to converge")
    assert int(freqs.sum()) == L and (freqs >= 1).all()
    return freqs.astype(np.int32)



# def build_tans_table(vocab: np.ndarray, counts: np.ndarray,
#                      table_log: int = TABLE_LOG):
#     """
#     Build tANS encode + decode tables for a given symbol distribution.

#     Parameters
#     ----------
#     vocab     : uint32 array, n_vocab unique symbols (sorted)
#     counts    : int64 array, co-indexed with vocab
#     table_log : log2 of table size

#     Returns
#     -------
#     dict with keys:
#         table_log   : int
#         vocab       : uint32 array (n_vocab,)
#         freqs       : int32 array (n_vocab,)   — quantised, sum=L
#         encode_ns   : int32 array (n_vocab,)   — next-state base per symbol
#         encode_nb   : uint8 array (n_vocab,)   — number of bits to flush per symbol
#         decode_sym  : uint32 array (L,)        — state → symbol
#         decode_ns   : int32 array  (L,)        — state → next state after decode
#     """
#     L         = 1 << table_log
#     n_vocab   = len(vocab)
#     freqs     = _quantise_freqs(counts, L)

#     # ── Step 1: spread symbols across L slots (Collet "fast spread") ──
#     # Each symbol s gets freqs[s] slots; slots are interleaved pseudo-randomly
#     # using the cumulative-freq offset trick so low-freq symbols are spread evenly.
#     slot        = np.empty(L, dtype=np.int32)   # slot[x] = symbol index
#     step        = (L >> 1) + (L >> 3) + 3       # must be coprime with L
#     pos         = 0
#     for si in range(n_vocab):
#         for _ in range(freqs[si]):
#             slot[pos % L] = si
#             pos += step

#     # ── Step 2: decode table: state x (in [L, 2L-1]) → (symbol, new_state) ──
#     # new_state[x] = freq[symbol] + rank_of_x_within_symbol_slots
#     # We use the spread order: symbol slot rank increases left-to-right in slot[]
#     decode_sym = vocab[slot]                          # uint32  (L,)
#     # For each slot position, its rank among same-symbol occurrences
#     # = cumcount within symbol group in slot order
#     decode_ns  = np.empty(L, dtype=np.int32)
#     sym_counter = np.zeros(n_vocab, dtype=np.int32)
#     for x in range(L):
#         si              = slot[x]
#         decode_ns[x]    = freqs[si] + sym_counter[si]   # in [L, 2L-1]
#         sym_counter[si] += 1

#     # ── Step 3: encode table: for each symbol, compute (nb, new_state_base) ──
#     # When encoding symbol s with current state x:
#     #   nb = table_log - floor(log2(x / freq[s]))  bits to flush
#     #   new_state = decode_ns mapped back from the symbol's slot
#     # We store per-symbol encode_ns (base next-state) and encode_nb arrays
#     # indexed by (state - freq[s]*2^k) but for vectorised encode we use:
#     #   nb(x,s)  = table_log + 1 - bit_length(x * freq[s] / L) ... simplified:
#     #   nb(x,s)  = table_log - floor(log2( floor(x * freq[s] / L + freq[s]) ))
#     #   Actually the standard formula: nb = (table_log + 1) - bit_length(step_x)
#     #   where step_x = x >> (bit_length(x) - table_log - 1 + (x >= freq[s]<<table_log))
#     # For simplicity and correctness we store the full (L × n_vocab) nb table
#     # only for n_vocab <= 65536; for larger vocabs we use the per-symbol formula.
#     # Here we store encode tables per-symbol (compact form used by vectorised encode).

#     # encode_nb[si]: number of bits to output before transition
#     # Specifically for state x: nb = nb_base[si] + (x >= thresh[si])
#     # where thresh[si] = (nb_base[si]+1 - table_log) ... Collet compact form:
#     #   nb_base[si] = table_log - floor(log2(freqs[si]))  ... approximately
#     # We store nb_base and thresh explicitly.
#     encode_nb_base = (table_log - np.floor(np.log2(freqs)).astype(np.int32)).astype(np.int32)
#     encode_thresh  = (freqs.astype(np.int64) << (encode_nb_base + 1)).astype(np.int64)
#     # new-state: after flushing nb bits from x, new state = decode_ns of the
#     # symbol slot whose rank matches.  We store encode_ns_base[si] = freqs[si]
#     # (decode_ns starts from freqs[si]+rank, so we need the mapping).
#     # For vectorised encode we precompute: for each symbol, a lookup from
#     # (x >> nb) → new_state.  But that requires a (n_vocab × L) table = too big.
#     # Instead we use the direct decode_ns inverse: store per-symbol slot list.
#     sym_slots = [[] for _ in range(n_vocab)]
#     for x in range(L):
#         sym_slots[slot[x]].append(decode_ns[x])
#     # sym_slots[si][rank] = new state after encoding the rank-th occurrence of si
#     # When encoding, rank within symbol = encode call counter per symbol.
#     # For VECTORISED encode we convert this to: new_state = sym_ns[si][x >> nb]
#     # i.e. the rank is determined by (x >> nb) - freqs[si] ... standard tANS.
#     # We pack sym_slots into a flat ragged array for IPC efficiency.
#     sym_ns_flat   = np.concatenate([np.array(ss, dtype=np.int32)
#                                     for ss in sym_slots])
#     sym_ns_offset = np.zeros(n_vocab + 1, dtype=np.int32)
#     for si, ss in enumerate(sym_slots):
#         sym_ns_offset[si + 1] = sym_ns_offset[si] + len(ss)

#     return {
#         "table_log":      table_log,
#         "vocab":          vocab.astype(np.uint32),
#         "freqs":          freqs,
#         "encode_nb_base": encode_nb_base,     # int32 (n_vocab,)
#         "encode_thresh":  encode_thresh,      # int64 (n_vocab,)
#         "sym_ns_flat":    sym_ns_flat,        # int32 (L,)
#         "sym_ns_offset":  sym_ns_offset,      # int32 (n_vocab+1,)
#         "decode_sym":     decode_sym,         # uint32 (L,)
#         "decode_ns":      decode_ns,          # int32 (L,)
#     }

def build_tans_table(vocab: np.ndarray, counts: np.ndarray) -> dict:
    """vocab: sorted unique symbols (uint16 or uint32). counts: co-indexed int64."""
    n_vocab   = len(vocab)
    table_log = choose_table_log(n_vocab)
    L         = 1 << table_log
    freqs     = _quantise_freqs(counts, L)
 
    step = (L >> 1) + (L >> 3) + 3
    slot = np.empty(L, dtype=np.int32)
    pos  = 0
    for si in range(n_vocab):
        f = int(freqs[si])
        for _ in range(f):
            slot[pos % L] = si
            pos += step
 
    decode_sym = vocab[slot].copy()
    decode_ns  = np.empty(L, dtype=np.int32)
    sym_ctr    = np.zeros(n_vocab, dtype=np.int32)
    for x in range(L):
        si = slot[x]
        decode_ns[x] = freqs[si] + sym_ctr[si]
        sym_ctr[si] += 1
 
    sym_slot_lists = [[] for _ in range(n_vocab)]
    for i in range(L):
        sym_slot_lists[slot[i]].append(i)
    sym_slot_flat = np.concatenate(
        [np.array(s, dtype=np.int32) for s in sym_slot_lists]
    ) if n_vocab > 0 else np.empty(0, dtype=np.int32)
    sym_slot_offset = np.zeros(n_vocab + 1, dtype=np.int32)
    for si in range(n_vocab):
        sym_slot_offset[si + 1] = sym_slot_offset[si] + len(sym_slot_lists[si])
 
    log2f   = np.floor(np.log2(freqs.astype(np.float64))).astype(np.int64)
    nb_base = (table_log - 1 - log2f).astype(np.int32)
    thresh  = (freqs.astype(np.int64) << (nb_base + 1)).astype(np.int64)
 
    return {
        "table_log": table_log,
        "vocab": vocab.copy(),
        "freqs": freqs,
        "nb_base": nb_base,
        "thresh": thresh,
        "sym_slot_flat": sym_slot_flat,
        "sym_slot_offset": sym_slot_offset,
        "decode_sym": decode_sym,
        "decode_ns": decode_ns,
    }



# ──────────────────────────────────────────────────────────────────────────────
# tANS ENCODE
# ──────────────────────────────────────────────────────────────────────────────

def tans_encode(symbols: np.ndarray, table: dict) -> bytes:
    """
    Encode a uint32 symbol array to a tANS bitstream (bytes).

    Algorithm
    ---------
    Process symbols RIGHT-TO-LEFT (standard ANS streaming order so decode
    goes left-to-right).  For each symbol s with current state x:
      1. nb = nb_base[s] + (x >= thresh[s])
      2. flush nb low bits of x into the bit buffer
      3. x = sym_ns[s][ (x >> nb) - freqs[s] ]

    Bit packing: bits are accumulated in a uint64 accumulator; flushed to
    output in 64-bit words when full.

    The final state x (in [L, 2L-1]) is stored as a 32-bit header so the
    decoder knows where to start.

    Returns raw bytes: [state:4B][bit_words:8B*...][padding_bits:1B]
    """
    table_log      = table["table_log"]
    L              = 1 << table_log
    vocab          = table["vocab"]
    freqs          = table["freqs"]
    nb_base        = table["encode_nb_base"]
    thresh         = table["encode_thresh"]
    sym_ns_flat    = table["sym_ns_flat"]
    sym_ns_offset  = table["sym_ns_offset"]

    n = len(symbols)
    if n == 0:
        return b"\x00" * 5  # empty stream

    # Build symbol → index map (vocab is sorted uint32)
    # Use searchsorted — vocab is sorted
    sym_idx = np.searchsorted(vocab.astype(np.int64),
                              symbols.astype(np.int64)).astype(np.int32)

    # Initial state: L (standard tANS starting state)
    x = L

    # Bit buffer
    bit_buf   = np.uint64(0)
    bit_count = 0
    words     = []           # list of uint64

    # Process RIGHT-TO-LEFT
    for i in range(n - 1, -1, -1):
        si  = sym_idx[i]
        nb  = int(nb_base[si]) + int(x >= thresh[si])
        # Flush nb bits (low bits of x)
        if nb > 0:
            bit_buf   |= np.uint64(x & ((1 << nb) - 1)) << np.uint64(bit_count)
            bit_count += nb
            while bit_count >= 64:
                words.append(int(bit_buf & np.uint64(0xFFFFFFFFFFFFFFFF)))
                bit_buf    = np.uint64(bit_buf >> np.uint64(64))
                bit_count -= 64
        # State transition
        rank = (x >> nb) - int(freqs[si])
        x    = int(sym_ns_flat[sym_ns_offset[si] + rank])

    # Flush remaining bits
    if bit_count > 0:
        words.append(int(bit_buf))

    # Pack: [final_state:uint32][n_words:uint32][padding_bits:uint8][words...]
    import struct
    header = struct.pack("<IIB", x, len(words), bit_count % 64 if bit_count % 64 != 0 else 0)
    body   = struct.pack(f"<{len(words)}Q", *words)
    return header + body


def _encode_chunk_py(sym_idx_l, freqs_l, nb_base_l, thresh_l,
                     slot_f_l, slot_o_l, L):
    n = len(sym_idx_l)
    if n == 0:
        return [], L, 0
    x = L
    accum = 0; accum_bits = 0; total_bits = 0; words_out = []
    for i in range(n - 1, -1, -1):
        si = sym_idx_l[i]
        nb = nb_base_l[si] + (1 if x >= thresh_l[si] else 0)
        accum |= (x & ((1 << nb) - 1)) << accum_bits
        accum_bits += nb; total_bits += nb
        while accum_bits >= 64:
            words_out.append(accum & 0xFFFFFFFFFFFFFFFF)
            accum >>= 64; accum_bits -= 64
        rank = (x >> nb) - freqs_l[si]
        x = L + slot_f_l[slot_o_l[si] + rank]
    if accum_bits > 0:
        words_out.append(accum & 0xFFFFFFFFFFFFFFFF)
    return words_out, x, total_bits
 
 
def _decode_chunk_py(words, total_bits, state, n_symbols, dec_sym_l, dec_ns_l, L):
    out = np.empty(n_symbols, dtype=np.int64)
    if n_symbols == 0:
        return out
    n_words = len(words)
    if total_bits == 0:
        cur_word, cur_bit = -1, -1
    else:
        last_bits = total_bits % 64
        cur_word = n_words - 1
        cur_bit  = 63 if last_bits == 0 else last_bits - 1
 
    x = state
    for i in range(n_symbols):
        sx = x - L
        x  = dec_ns_l[sx]
        out[i] = dec_sym_l[sx]
        while x < L:
            if cur_word < 0:
                bit = 0
            else:
                bit = (words[cur_word] >> cur_bit) & 1
                cur_bit -= 1
                if cur_bit < 0:
                    cur_word -= 1; cur_bit = 63
            x = (x << 1) | bit
    return out


def _encode_chunk_nb(sym_idx, freqs, nb_base, thresh, slot_flat, slot_off, L):
    n = sym_idx.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.uint64), np.int64(L), np.int64(0)
 
    max_words = (n * 18) // 64 + 4   # TABLE_LOG_MAX=16 → <=17 bits/sym worst case
    words = np.empty(max_words, dtype=np.uint64)
    word_count = 0
 
    accum = np.uint64(0)
    accum_bits = 0
    total_bits = np.int64(0)
    x = np.int64(L)
 
    for ii in range(n - 1, -1, -1):
        si = sym_idx[ii]
        f  = np.int64(freqs[si])
        nb = nb_base[si] + (1 if x >= thresh[si] else 0)
        bits_val = x & ((1 << nb) - 1)
        accum |= np.uint64(bits_val) << np.uint64(accum_bits)
        accum_bits += nb
        total_bits += nb
        while accum_bits >= 64:
            words[word_count] = accum
            word_count += 1
            accum = np.uint64(0)
            accum_bits -= 64
        rank = (x >> nb) - f
        x = L + slot_flat[slot_off[si] + rank]
 
    if accum_bits > 0:
        words[word_count] = accum
        word_count += 1
 
    return words[:word_count].copy(), x, total_bits
 
 
@njit(cache=True, nogil=True)
def _decode_chunk_nb(words, total_bits, state, n_symbols, decode_sym, decode_ns, L):
    out = np.empty(n_symbols, dtype=np.int64)
    if n_symbols == 0:
        return out
    n_words = words.shape[0]
    if total_bits == 0:
        cur_word, cur_bit = -1, -1
    else:
        last_bits = total_bits % 64
        cur_word = n_words - 1
        cur_bit  = 63 if last_bits == 0 else last_bits - 1
 
    x = state
    for i in range(n_symbols):
        sx = x - L
        x  = decode_ns[sx]
        out[i] = decode_sym[sx]
        while x < L:
            if cur_word < 0:
                bit = 0
            else:
                bit = int((words[cur_word] >> np.uint64(cur_bit)) & np.uint64(1))
                cur_bit -= 1
                if cur_bit < 0:
                    cur_word -= 1
                    cur_bit = 63
            x = (x << 1) | bit
    return out
 
 
# Warm up JIT once at import time (tiny dummy call) so first real call isn't slow,
# and so any compile error surfaces immediately instead of mid-pipeline.
if HAVE_NUMBA:
    try:
        _w, _s, _b = _encode_chunk_nb(
            np.array([0, 0], dtype=np.int32),
            np.array([2], dtype=np.int32),
            np.array([0], dtype=np.int32),
            np.array([4], dtype=np.int64),
            np.array([0, 1], dtype=np.int32),
            np.array([0, 2], dtype=np.int32),
            4,
        )
        _decode_chunk_nb(_w, _b, _s, 2,
                        np.array([0, 0, 0, 0], dtype=np.int64),
                        np.array([2, 3, 2, 3], dtype=np.int64), 4)
    except Exception:
        HAVE_NUMBA = False
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CHUNKED ENCODE / DECODE (public API)
# ─────────────────────────────────────────────────────────────────────────────
 
# Each chunk independently encodes CHUNK_SYMS symbols starting from state L,
# under the SAME table. Chunks are fully independent → safe to run concurrently.
CHUNK_SYMS = 2_000_000
 
 
def tans_encode_stream(symbols: np.ndarray, table: dict, executor=None) -> bytes:
    """
    Encode `symbols` (already mapped to vocab via the table, or raw — we map
    here) into a chunked tANS bitstream.
 
    Wire format
    -----------
    [n_chunks: u32]
    For each chunk:
        [n_symbols: u32][state: i64][total_bits: i64][n_words: u32]
        [n_words x u64 words]
 
    `executor`: optional ThreadPoolExecutor. If numba is available, chunk
    encodes release the GIL and run truly in parallel; if numba is not
    available, chunks are still independent but run sequentially in the
    calling thread (an executor is accepted but not required for correctness).
    """
    n = len(symbols)
    L = 1 << table["table_log"]
    vocab = table["vocab"]
 
    if n == 0:
        return struct.pack("<I", 0)
 
    sym_idx = np.searchsorted(vocab.astype(np.int64),
                              symbols.astype(np.int64)).astype(np.int32)
 
    n_chunks = (n + CHUNK_SYMS - 1) // CHUNK_SYMS
    chunks = [sym_idx[i * CHUNK_SYMS:(i + 1) * CHUNK_SYMS] for i in range(n_chunks)]
 
    freqs   = table["freqs"]
    nb_base = table["nb_base"]
    thresh  = table["thresh"]
    slot_f  = table["sym_slot_flat"]
    slot_o  = table["sym_slot_offset"]
 
    if HAVE_NUMBA:
        def _enc(c):
            return _encode_chunk_nb(c, freqs, nb_base, thresh, slot_f, slot_o, L)
    else:
        freqs_l, nb_base_l, thresh_l = freqs.tolist(), nb_base.tolist(), thresh.tolist()
        slot_f_l, slot_o_l = slot_f.tolist(), slot_o.tolist()
        def _enc(c):
            w, s, b = _encode_chunk_py(c.tolist(), freqs_l, nb_base_l, thresh_l,
                                       slot_f_l, slot_o_l, L)
            return np.array(w, dtype=np.uint64), s, b
 
    if executor is not None and HAVE_NUMBA and len(chunks) > 1:
        results = list(executor.map(_enc, chunks))
    else:
        results = [_enc(c) for c in chunks]
 
    # ── Gather: build output buffer in deterministic chunk order ──
    out = [struct.pack("<I", n_chunks)]
    for chunk, (words, state, total_bits) in zip(chunks, results):
        words_arr = np.asarray(words, dtype=np.uint64)
        out.append(struct.pack("<IqqI", len(chunk), int(state), int(total_bits),
                               len(words_arr)))
        out.append(words_arr.tobytes())
    return b"".join(out)
 
 
def tans_decode_stream(data: bytes, table: dict, executor=None) -> np.ndarray:
    """Inverse of tans_encode_stream. Returns the original symbol dtype array."""
    L = 1 << table["table_log"]
    decode_sym = table["decode_sym"]
    decode_ns  = table["decode_ns"]
    out_dtype  = decode_sym.dtype
 
    (n_chunks,) = struct.unpack_from("<I", data, 0)
    if n_chunks == 0:
        return np.empty(0, dtype=out_dtype)
 
    offset = 4
    chunk_specs = []
    for _ in range(n_chunks):
        n_sym, state, total_bits, n_words = struct.unpack_from("<IqqI", data, offset)
        offset += 24
        words = np.frombuffer(data, dtype=np.uint64, count=n_words, offset=offset)
        offset += 8 * n_words
        chunk_specs.append((n_sym, state, total_bits, words))
 
    decode_sym_i64 = decode_sym.astype(np.int64)
    decode_ns_i64  = decode_ns.astype(np.int64)
 
    if HAVE_NUMBA:
        def _dec(spec):
            n_sym, state, total_bits, words = spec
            return _decode_chunk_nb(words, total_bits, state, n_sym,
                                    decode_sym_i64, decode_ns_i64, L)
    else:
        dec_sym_l = decode_sym_i64.tolist()
        dec_ns_l  = decode_ns_i64.tolist()
        def _dec(spec):
            n_sym, state, total_bits, words = spec
            return _decode_chunk_py(words.tolist(), total_bits, state, n_sym,
                                    dec_sym_l, dec_ns_l, L)
 
    if executor is not None and HAVE_NUMBA and len(chunk_specs) > 1:
        results = list(executor.map(_dec, chunk_specs))
    else:
        results = [_dec(s) for s in chunk_specs]
 
    # ── Merge: concatenate in chunk order, cast back to symbol dtype ──
    merged = np.concatenate(results) if len(results) > 1 else results[0]
    return merged.astype(out_dtype)
 




def tans_decode(data: bytes, n_symbols: int, table: dict) -> np.ndarray:
    """
    Decode a tANS bitstream back to the original uint32 symbol array.

    Reads bits LEFT-TO-RIGHT (reverse of encode direction), reconstructing
    symbols in forward order.

    Parameters
    ----------
    data      : bytes produced by tans_encode
    n_symbols : number of symbols to decode (stored separately in metadata)
    table     : same dict returned by build_tans_table

    Returns
    -------
    uint32 array of length n_symbols
    """
    import struct

    table_log  = table["table_log"]
    L          = 1 << table_log
    decode_sym = table["decode_sym"]
    decode_ns  = table["decode_ns"]

    if n_symbols == 0:
        return np.empty(0, dtype=np.uint32)

    # Parse header
    state, n_words, last_bits = struct.unpack_from("<IIB", data, 0)
    offset = 9  # 4+4+1

    words = struct.unpack_from(f"<{n_words}Q", data, offset)
    words = list(words)

    # Read bits LEFT-TO-RIGHT by consuming words from the END
    # (encode wrote right-to-left, so the first-encoded bits are at the end)
    words.reverse()
    bit_buf   = np.uint64(0)
    bit_count = 0

    def _refill():
        nonlocal bit_buf, bit_count
        if words:
            w = words.pop()
            bit_buf   |= np.uint64(w) << np.uint64(bit_count)
            bit_count += 64

    x = state
    out = np.empty(n_symbols, dtype=np.uint32)

    for i in range(n_symbols):
        # State x is in [L, 2L-1]; decode
        sx = x - L
        s  = decode_sym[sx]
        x  = decode_ns[sx]
        out[i] = s

        # Read bits to normalise x back into [L, 2L-1]
        nb = 0
        # x is now in [freqs[sym_idx], 2*freqs[sym_idx]-1] after decode_ns
        # normalise: while x < L → read 1 bit and shift
        while x < L:
            if bit_count == 0:
                _refill()
            if bit_count > 0:
                x = (x << 1) | int(bit_buf & np.uint64(1))
                bit_buf    >>= np.uint64(1)
                bit_count  -= 1
                nb += 1

    return out


# ──────────────────────────────────────────────────────────────────────────────
# FAST VECTORISED ENCODE (numpy-accelerated inner loop)
# ──────────────────────────────────────────────────────────────────────────────

def tans_encode_fast(symbols: np.ndarray, table: dict) -> bytes:
    """
    Faster tANS encode using numpy for symbol-index lookup and batch bit ops.

    For large tensors (millions of symbols) the Python loop in tans_encode is
    too slow.  This version:
      - Maps all symbols → indices with one searchsorted call
      - Processes symbols in chunks of CHUNK symbols using numpy
      - Accumulates bits into a pre-allocated uint64 word array

    The inner per-symbol transition loop cannot be fully vectorised (it is
    inherently sequential: each state depends on the previous), but all
    ancillary work (lookup, bit masking) is batched.
    """
    import struct

    table_log      = table["table_log"]
    L              = 1 << table_log
    vocab          = table["vocab"]
    freqs          = table["freqs"]
    nb_base        = table["encode_nb_base"]
    thresh         = table["encode_thresh"]
    sym_ns_flat    = table["sym_ns_flat"]
    sym_ns_offset  = table["sym_ns_offset"]

    n = len(symbols)
    if n == 0:
        return struct.pack("<IIB", L, 0, 0)

    # Map symbols → indices (one vectorised call)
    sym_idx = np.searchsorted(vocab.astype(np.int64),
                              symbols.astype(np.int64)).astype(np.int32)

    # Pre-fetch per-symbol arrays into local numpy arrays for fast indexing
    nb_base_  = nb_base     # int32 (n_vocab,)
    thresh_   = thresh      # int64 (n_vocab,)
    freqs_    = freqs       # int32 (n_vocab,)
    ns_flat_  = sym_ns_flat # int32 (L,)
    ns_off_   = sym_ns_offset # int32 (n_vocab+1,)

    # Estimate output size: upper bound = n_symbols * (table_log+1) bits
    est_words = (n * (table_log + 2) + 63) // 64 + 4
    word_buf  = np.zeros(est_words, dtype=np.uint64)
    word_pos  = 0
    bit_buf   = np.uint64(0)
    bit_count = 0

    x = L  # initial state

    # Process RIGHT-TO-LEFT in python loop (state is sequential)
    # Pull out local python ints for tightest inner loop
    nb_base_list  = nb_base_.tolist()
    thresh_list   = thresh_.tolist()
    freqs_list    = freqs_.tolist()
    ns_flat_list  = ns_flat_.tolist()
    ns_off_list   = ns_off_.tolist()
    sym_idx_list  = sym_idx.tolist()

    words_out = []

    for i in range(n - 1, -1, -1):
        si  = sym_idx_list[i]
        nb  = nb_base_list[si] + (x >= thresh_list[si])
        if nb > 0:
            # Flush nb low bits
            bits_val    = x & ((1 << nb) - 1)
            bit_buf    |= np.uint64(bits_val) << np.uint64(bit_count)
            bit_count  += nb
            while bit_count >= 64:
                words_out.append(int(bit_buf & np.uint64(0xFFFFFFFFFFFFFFFF)))
                bit_buf    = np.uint64(bit_buf >> np.uint64(64))
                bit_count -= 64
        rank = (x >> nb) - freqs_list[si]
        x    = ns_flat_list[ns_off_list[si] + rank]

    if bit_count > 0:
        words_out.append(int(bit_buf))

    header = struct.pack("<IIB", x, len(words_out), bit_count % 64 if bit_count % 64 != 0 else 0)
    body   = struct.pack(f"<{len(words_out)}Q", *words_out)
    return header + body


# ──────────────────────────────────────────────────────────────────────────────
# PER-TENSOR WORKER  (runs in a subprocess)
# ──────────────────────────────────────────────────────────────────────────────

def _encode_tensor_worker(args):
    """
    Worker: build per-tensor tANS table and encode one tensor.

    Input  : (tensor_idx, symbols_uint32, verify)
    Output : (tensor_idx, compressed_bytes, table_dict, n_symbols, original_nbytes)
    """
    tensor_idx, symbols, verify = args

    vocab, inverse, counts = np.unique(symbols, return_inverse=True,
                                       return_counts=True)
    table   = build_tans_table(vocab, counts)
    stream  = tans_encode_fast(symbols, table)

    if verify:
        decoded = tans_decode(stream, len(symbols), table)
        if not np.array_equal(decoded, symbols):
            raise RuntimeError(
                f"[Tensor {tensor_idx}] tANS roundtrip FAILED: "
                f"first mismatch at {np.where(decoded != symbols)[0][0]}"
            )

    # Strip raw symbol arrays from table before returning over IPC
    # (they can be reconstructed from vocab+counts on decode side)
    compact_table = {k: v for k, v in table.items()
                     if k not in ("sym_ns_flat", "sym_ns_offset")}
    # Keep full table — needed for decode
    return tensor_idx, stream, table, len(symbols), int(symbols.nbytes)


# ──────────────────────────────────────────────────────────────────────────────
# CLUSTER BUILD + ENCODE WORKER
# ──────────────────────────────────────────────────────────────────────────────

def _build_cluster_tables_worker(args):
    """
    Worker: given a list of (tensor_idx, symbols) for one cluster,
    merge their vocab/counts, build ONE shared tANS table, encode all tensors.

    Input  : (cluster_id, list of (tensor_idx, symbols_uint32), verify)
    Output : (cluster_id,
              list of (tensor_idx, compressed_bytes, n_symbols, orig_nbytes),
              shared_table)
    """
    cluster_id, tensor_list, verify = args

    # Merge vocab/counts from all tensors in cluster
    all_vocab, all_counts = [], []
    for _, sym in tensor_list:
        v, c = np.unique(sym, return_counts=True)
        all_vocab.append(v.astype(np.int64))
        all_counts.append(c)

    merged_v = np.concatenate(all_vocab)
    merged_c = np.concatenate(all_counts)
    order    = np.argsort(merged_v, kind="stable")
    sv, sc   = merged_v[order], merged_c[order]
    change   = np.concatenate([[True], sv[1:] != sv[:-1]])
    u_vocab  = sv[change].astype(np.uint32)
    u_counts = np.add.reduceat(sc, np.where(change)[0]).astype(np.int64)

    table = build_tans_table(u_vocab, u_counts)

    results = []
    for tensor_idx, sym in tensor_list:
        stream = tans_encode_fast(sym, table)
        if verify:
            decoded = tans_decode(stream, len(sym), table)
            if not np.array_equal(decoded, sym):
                raise RuntimeError(
                    f"[Cluster {cluster_id} Tensor {tensor_idx}] roundtrip FAILED"
                )
        results.append((tensor_idx, stream, len(sym), int(sym.nbytes)))

    return cluster_id, results, table


# ──────────────────────────────────────────────────────────────────────────────
# SHARD LOADER
# ──────────────────────────────────────────────────────────────────────────────

def _load_optim_shard(rank: int, checkpoint_dir: str):
    path = os.path.join(
        checkpoint_dir,
        f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt"
    )
    print(f"  Loading rank {rank}: {path}", flush=True)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    opt  = ckpt["optimizer_state_dict"]

    weight_flat = opt["fp32_flat_groups"][0]

    state = opt["optimizer_state_dict"]["state"]
    ea_list, easq_list = [], []
    for v in state.values():
        if "exp_avg"    in v: ea_list.append(v["exp_avg"].flatten())
        if "exp_avg_sq" in v: easq_list.append(v["exp_avg_sq"].flatten())

    exp_avg    = torch.cat(ea_list)
    exp_avg_sq = torch.cat(easq_list)

    # Also save the full ckpt dict structure (without tensors) for reconstruct
    # We need param_shapes from model state
    return rank, weight_flat, exp_avg, exp_avg_sq, ckpt


def load_shards_parallel(checkpoint_dir: str, num_ranks: int):
    print("Loading all optimizer shards in parallel...")
    results = [None] * num_ranks
    with ThreadPoolExecutor(max_workers=num_ranks) as pool:
        futures = {
            pool.submit(_load_optim_shard, r, checkpoint_dir): r
            for r in range(num_ranks)
        }
        for fut in as_completed(futures):
            r, w, ea, easq, ckpt = fut.result()
            results[r] = (w, ea, easq, ckpt)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# TENSOR EXTRACTION  (from flat buffer using param_shapes)
# ──────────────────────────────────────────────────────────────────────────────

def extract_uint32_tensors(flat: torch.Tensor, param_shapes) -> list[np.ndarray]:
    """Split flat fp32 buffer by param_shapes → list of uint32 numpy arrays."""
    f32_np    = flat.detach().cpu().float().numpy()
    uint32_np = f32_np.view(np.uint32)
    tensors   = []
    offset    = 0
    for shape in param_shapes.values():
        numel = shape.numel()
        tensors.append(uint32_np[offset:offset + numel].copy())
        offset += numel
    return tensors


# ──────────────────────────────────────────────────────────────────────────────
# CLUSTERING
# ──────────────────────────────────────────────────────────────────────────────

def cluster_tensors(tensors: list[np.ndarray]) -> dict[int, list[int]]:
    """
    Cluster tensors by (unique_ratio, entropy) features using MiniBatchKMeans.
    Returns {cluster_id: [tensor_indices]}.
    """
    features = []
    for sym in tensors:
        vocab, counts = np.unique(sym, return_counts=True)
        probs = counts / counts.sum()
        ent   = float(-np.dot(probs, np.log2(probs + 1e-300)))
        u_r   = len(vocab) / len(sym)
        features.append([u_r, ent])
    F         = np.array(features, dtype=np.float32)
    n_clust   = max(8, len(tensors) // 40)
    labels    = MiniBatchKMeans(n_clusters=n_clust, random_state=0, n_init="auto",
                                batch_size=min(1024, len(tensors))).fit_predict(F)
    clusters  = defaultdict(list)
    for i, c in enumerate(labels):
        clusters[int(c)].append(i)
    return dict(clusters)


# ──────────────────────────────────────────────────────────────────────────────
# CHECKSUM
# ──────────────────────────────────────────────────────────────────────────────

def sha256_tensor(t: torch.Tensor) -> str:
    arr = t.detach().cpu().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────────────
# SAVE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def save_compressed(out_dir: str, tag: str,
                    streams: list[bytes],
                    tables:  list[dict],
                    n_symbols_list: list[int],
                    shapes: list,
                    strategy: str,
                    cluster_assignment: dict | None = None):
    """
    Save compressed bitstreams + all metadata needed for decode.

    Layout:
        <out_dir>/<strategy>/<tag>.bin        — concatenated bitstreams
        <out_dir>/<strategy>/<tag>_meta.pkl   — tables, offsets, n_symbols, shapes
    """
    strat_dir = os.path.join(out_dir, strategy)
    os.makedirs(strat_dir, exist_ok=True)

    # Pack all per-tensor streams into one file with an offset table
    offsets = [0]
    for s in streams:
        offsets.append(offsets[-1] + len(s))

    bin_path  = os.path.join(strat_dir, f"{tag}.bin")
    meta_path = os.path.join(strat_dir, f"{tag}_meta.pkl")

    with open(bin_path, "wb") as f:
        for s in streams:
            f.write(s)

    meta = {
        "strategy":          strategy,
        "tag":               tag,
        "n_tensors":         len(streams),
        "offsets":           offsets,          # byte offsets into .bin
        "n_symbols_list":    n_symbols_list,   # symbols per tensor
        "shapes":            shapes,           # original param shapes
        "tables":            tables,           # one table per tensor OR per cluster
        "cluster_assignment": cluster_assignment,  # {tensor_idx: cluster_id} or None
    }
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

    return bin_path, meta_path


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load model checkpoint to get param_shapes ──
    model_ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, "zero_pp_rank_0_mp_rank_00_model_states.pt"),
        map_location="cpu", weights_only=False,
    )
    param_shapes = model_ckpt["param_shapes"][0]
    print(f"param_shapes: {len(param_shapes)} tensors")

    # ── Load all optimizer shards ──
    shard_data = load_shards_parallel(CHECKPOINT_DIR, NUM_RANKS)

    # ── Compute checksums of original files ──
    print("\nComputing original file checksums...")
    orig_checksums = {}
    for r in range(NUM_RANKS):
        p = os.path.join(CHECKPOINT_DIR,
                         f"bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt")
        orig_checksums[f"rank{r}_optim"] = sha256_file(p)
        print(f"  rank {r}: {orig_checksums[f'rank{r}_optim'][:16]}...")

    report_lines = []
    report_lines.append("=" * 72)
    report_lines.append("tANS CHECKPOINT COMPRESSION REPORT")
    report_lines.append("=" * 72)

    grand_raw   = 0
    grand_pt    = {"stream": 0, "tables": 0}
    grand_cl    = {"stream": 0, "tables": 0}

    STREAM_NAMES = ["weights", "exp_avg", "exp_avg_sq"]

    for rank, (weight_flat, exp_avg, exp_avg_sq, ckpt) in enumerate(shard_data):
        flat_streams = {
            "weights":    weight_flat,
            "exp_avg":    exp_avg,
            "exp_avg_sq": exp_avg_sq,
        }

        report_lines.append(f"\n{'─'*72}")
        report_lines.append(f"RANK {rank}")
        report_lines.append(f"{'─'*72}")

        for sname in STREAM_NAMES:
            flat = flat_streams[sname]
            tensors = extract_uint32_tensors(flat, param_shapes)
            n_tensors = len(tensors)
            raw_bytes = sum(t.nbytes for t in tensors)
            grand_raw += raw_bytes

            print(f"\n[Rank {rank} | {sname}] {n_tensors} tensors, "
                  f"{raw_bytes/1e9:.3f} GB raw", flush=True)

            tag = f"rank{rank}_{sname}"

            # ── (A) PER-TENSOR STRATEGY ──────────────────────────────────────
            print(f"  [A] Per-tensor encode ({n_tensors} workers)...", flush=True)
            t_pt_start = time.time()

            pt_tasks = [
                (i, tensors[i], VERIFY_DECODE)
                for i in range(n_tensors)
            ]
            pt_streams   = [None] * n_tensors
            pt_tables    = [None] * n_tensors
            pt_n_symbols = [None] * n_tensors

            # Use ProcessPoolExecutor; chunk to avoid memory explosion
            batch = max(1, n_tensors // N_WORKERS)
            with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
                futs = {pool.submit(_encode_tensor_worker, task): task[0]
                        for task in pt_tasks}
                done = 0
                for fut in as_completed(futs):
                    tidx, stream, table, nsym, _ = fut.result()
                    pt_streams[tidx]   = stream
                    pt_tables[tidx]    = table
                    pt_n_symbols[tidx] = nsym
                    done += 1
                    if done % 50 == 0 or done == n_tensors:
                        print(f"    {done}/{n_tensors}", end="\r", flush=True)

            t_pt = time.time() - t_pt_start
            pt_stream_bytes = sum(len(s) for s in pt_streams)
            pt_table_bytes  = sum(
                sum(v.nbytes if isinstance(v, np.ndarray) else 0
                    for v in tbl.values())
                for tbl in pt_tables
            )
            grand_pt["stream"] += pt_stream_bytes
            grand_pt["tables"] += pt_table_bytes

            bin_pt, meta_pt = save_compressed(
                OUT_DIR, tag, pt_streams, pt_tables,
                pt_n_symbols,
                list(param_shapes.keys()),
                "per_tensor",
            )
            print(f"  [A] Done in {t_pt:.1f}s  "
                  f"stream={pt_stream_bytes/1e6:.1f} MB  "
                  f"tables={pt_table_bytes/1e6:.1f} MB", flush=True)

            # ── (B) CLUSTERED STRATEGY ────────────────────────────────────────
            print(f"  [B] Clustering {n_tensors} tensors...", flush=True)
            t_cl_start = time.time()

            clusters = cluster_tensors(tensors)
            cluster_groups = list(clusters.items())   # (cid, [idxs])
            print(f"      {len(cluster_groups)} clusters", flush=True)

            cl_tasks = [
                (cid, [(i, tensors[i]) for i in idxs], VERIFY_DECODE)
                for cid, idxs in cluster_groups
            ]
            # One worker per cluster
            cl_results_map = {}  # cluster_id → (results, table)
            with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(cl_tasks))) as pool:
                futs = {pool.submit(_build_cluster_tables_worker, task): task[0]
                        for task in cl_tasks}
                done = 0
                for fut in as_completed(futs):
                    cid, results, table = fut.result()
                    cl_results_map[cid] = (results, table)
                    done += 1
                    if done % 4 == 0 or done == len(cl_tasks):
                        print(f"    clusters {done}/{len(cl_tasks)}", end="\r", flush=True)

            # Reassemble in tensor order
            cl_streams   = [None] * n_tensors
            cl_tables    = [None] * n_tensors   # one per tensor but shared obj
            cl_n_symbols = [None] * n_tensors
            cluster_assignment = {}             # tensor_idx → cluster_id

            cluster_table_map = {}              # cid → table (deduplicated for storage)
            for cid, idxs in cluster_groups:
                results, table = cl_results_map[cid]
                cluster_table_map[cid] = table
                for tidx, stream, nsym, _ in results:
                    cl_streams[tidx]   = stream
                    cl_tables[tidx]    = table   # same object for whole cluster
                    cl_n_symbols[tidx] = nsym
                    cluster_assignment[tidx] = cid

            # Table bytes: count each unique table ONCE
            cl_table_bytes = sum(
                sum(v.nbytes if isinstance(v, np.ndarray) else 0
                    for v in tbl.values())
                for tbl in cluster_table_map.values()
            )
            t_cl = time.time() - t_cl_start
            cl_stream_bytes = sum(len(s) for s in cl_streams)
            grand_cl["stream"] += cl_stream_bytes
            grand_cl["tables"] += cl_table_bytes

            # For save: store one table per cluster (not per tensor)
            cl_tables_unique = [cluster_table_map[cid]
                                 for cid, _ in cluster_groups]
            bin_cl, meta_cl = save_compressed(
                OUT_DIR, tag, cl_streams, cl_tables_unique,
                cl_n_symbols,
                list(param_shapes.keys()),
                "clustered",
                cluster_assignment=cluster_assignment,
            )
            print(f"  [B] Done in {t_cl:.1f}s  "
                  f"stream={cl_stream_bytes/1e6:.1f} MB  "
                  f"tables={cl_table_bytes/1e6:.1f} MB", flush=True)

            # ── Per-stream report section ──────────────────────────────────────
            def pct(v): return f"{v / raw_bytes * 100:.2f}%"

            report_lines.append(f"\n  Stream: {sname}  |  raw={raw_bytes/1e6:.1f} MB")
            report_lines.append(f"  {'Metric':<30} {'Per-Tensor':>14} {'Clustered':>14}")
            report_lines.append(f"  {'-'*58}")
            report_lines.append(f"  {'Compressed stream (MB)':<30} "
                                 f"{pt_stream_bytes/1e6:>14.1f} "
                                 f"{cl_stream_bytes/1e6:>14.1f}")
            report_lines.append(f"  {'Table/metadata (MB)':<30} "
                                 f"{pt_table_bytes/1e6:>14.1f} "
                                 f"{cl_table_bytes/1e6:>14.1f}")
            report_lines.append(f"  {'Total (MB)':<30} "
                                 f"{(pt_stream_bytes+pt_table_bytes)/1e6:>14.1f} "
                                 f"{(cl_stream_bytes+cl_table_bytes)/1e6:>14.1f}")
            report_lines.append(f"  {'Stream ratio':<30} "
                                 f"{pct(pt_stream_bytes):>14} "
                                 f"{pct(cl_stream_bytes):>14}")
            report_lines.append(f"  {'Total ratio':<30} "
                                 f"{pct(pt_stream_bytes+pt_table_bytes):>14} "
                                 f"{pct(cl_stream_bytes+cl_table_bytes):>14}")
            report_lines.append(f"  {'Encode time (s)':<30} "
                                 f"{t_pt:>14.1f} "
                                 f"{t_cl:>14.1f}")
            report_lines.append(f"  {'# tables':<30} "
                                 f"{n_tensors:>14} "
                                 f"{len(cluster_groups):>14}")

    # ── Grand totals ──────────────────────────────────────────────────────────
    report_lines.append(f"\n{'='*72}")
    report_lines.append("GRAND TOTALS (all ranks, all streams)")
    report_lines.append(f"{'='*72}")
    report_lines.append(f"  Raw data:               {grand_raw/1e9:.3f} GB")

    for label, d in [("Per-Tensor", grand_pt), ("Clustered", grand_cl)]:
        total = d["stream"] + d["tables"]
        report_lines.append(f"\n  Strategy: {label}")
        report_lines.append(f"    Stream:   {d['stream']/1e9:.3f} GB  "
                             f"({d['stream']/grand_raw*100:.2f}%)")
        report_lines.append(f"    Tables:   {d['tables']/1e9:.3f} GB  "
                             f"({d['tables']/grand_raw*100:.2f}%)")
        report_lines.append(f"    Total:    {total/1e9:.3f} GB  "
                             f"({total/grand_raw*100:.2f}%)")
        report_lines.append(f"    Savings:  {(grand_raw - total)/1e9:.3f} GB  "
                             f"({(1 - total/grand_raw)*100:.2f}%)")

    report_lines.append(f"\n  Clustering vs Per-Tensor stream delta: "
                         f"{(grand_cl['stream'] - grand_pt['stream'])/1e6:+.1f} MB")
    report_lines.append(f"  Clustering vs Per-Tensor table delta:  "
                         f"{(grand_cl['tables'] - grand_pt['tables'])/1e6:+.1f} MB")
    report_lines.append(f"  Clustering vs Per-Tensor total delta:  "
                         f"{(grand_cl['stream']+grand_cl['tables'] - grand_pt['stream'] - grand_pt['tables'])/1e6:+.1f} MB")

    report_lines.append(f"\n  Total wall time: {time.time()-t0:.1f} s")

    # ── Save original checksums ────────────────────────────────────────────────
    cksum_path = os.path.join(OUT_DIR, "original_checksums.pkl")
    with open(cksum_path, "wb") as f:
        pickle.dump(orig_checksums, f)
    report_lines.append(f"\n  Original checksums saved to: {cksum_path}")

    # ── Write report ──────────────────────────────────────────────────────────
    report_text = "\n".join(report_lines)
    report_path = os.path.join(OUT_DIR, "report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)

    print("\n" + report_text)
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()