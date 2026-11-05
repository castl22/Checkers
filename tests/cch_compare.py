#!/usr/bin/env python3
"""
cch_compare.py -- are two .cch stream files the same checkpoint, tensor by tensor?

    python cch_compare.py A.cch B.cch        exit 0 = same, 1 = different

The streaming writer (2026-11-05) appends tensors in the order the workers
finish them, so two saves of the same state can place the same bytes at
different offsets: whole-file checksums no longer compare saves.  This compares
what matters instead: the same tensor names; per tensor the same dtype, codec,
flags, shape, codec parameters and identical payload and aux bytes; and the
same header fields apart from the layout offsets.  Reads v2 and v3 files.
"""
import struct, sys

HDR = struct.Struct("<8I d 6Q")
IDX = 432


def load(path):
    b = open(path, "rb").read()
    h = HDR.unpack_from(b, 0)
    n, idx_off, data_off = h[6], h[9], h[10]
    ents = {}
    for k in range(n):
        e = b[idx_off + IDX * k: idx_off + IDX * (k + 1)]
        name = e[:256].split(b"\0", 1)[0].decode()
        do, ds, ao, asz = struct.unpack_from("<4Q", e, 344)
        meta = e[256:344] + e[376:IDX]        # dtype/codec/flags/shape + codec params
        ents[name] = (meta, b[data_off + do: data_off + do + ds], b[data_off + ao: data_off + ao + asz])
    # header minus index_offset / data_offset (layout), keep the rest
    return h[:9] + h[11:13], ents


def main():
    (ha, ea), (hb, eb) = load(sys.argv[1]), load(sys.argv[2])
    probs = []
    if ha != hb:
        probs.append(f"header differs: {ha} vs {hb}")
    if set(ea) != set(eb):
        probs.append(f"tensor names differ: only in A {sorted(set(ea) - set(eb))[:3]}, "
                     f"only in B {sorted(set(eb) - set(ea))[:3]}")
    for name in sorted(set(ea) & set(eb)):
        for what, x, y in zip(("metadata", "payload", "aux"), ea[name], eb[name]):
            if x != y:
                probs.append(f"{name}: {what} differs")
    if probs:
        print("DIFFERENT: " + "; ".join(probs[:5]))
        sys.exit(1)
    print(f"same: {len(ea)} tensors")


if __name__ == "__main__":
    main()
