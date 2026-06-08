#!/usr/bin/env python3

import re
from collections import defaultdict
import sys

totals = defaultdict(int)

with open(sys.argv[1]) as f:
    for line in f:
        m = re.search(r'bytes=(\d+)', line)
        if not m:
            continue

        b = int(m.group(1))

        if '[master_weights::' in line:
            totals['master_weights'] += b
        elif '[optimizer.exp_avg_sq::' in line:
            totals['exp_avg_sq'] += b
        elif '[optimizer.exp_avg::' in line:
            totals['exp_avg'] += b
        else:
            totals['model_weights'] += b

grand_total = sum(totals.values())

for k, v in totals.items():
    print(f"{k:15s} {v/1024**3:10.3f} GiB")

print("-" * 40)
print(f"TOTAL           {grand_total/1024**3:10.3f} GiB")