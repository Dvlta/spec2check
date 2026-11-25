"""Disagreement between reference readings of each ambiguous criterion as the simulated pore size changes.

The paper's population uses a median pore size (size_scale) of 0.12 mm; this repeats the comparison at 0.10 and 0.15 mm.
"""
import itertools
import random

from bench import make_part, specs


def population(size_scale, n=4000):
    rng = random.Random(7)
    return [make_part(rng, f"p{i}", n_mean=10, size_scale=size_scale, cluster_p=0.3, scale=rng.choice([1.0, 0.5])) for i in range(n)]


def main():
    T4 = [s for s in specs() if s.tier == "T4"]
    scales = (0.10, 0.12, 0.15)
    pops = {sc: population(sc) for sc in scales}
    print("criterion  " + "  ".join(f"max@{sc:.2f}" for sc in scales))
    for s in T4:
        cells = []
        for sc in scales:
            v = {k: [f(p) for p in pops[sc]] for k, f in s.refs.items()}
            worst = max(sum(a != b for a, b in zip(v[x], v[y])) for x, y in itertools.combinations(v, 2))
            cells.append(f"{worst / len(pops[sc]):8.3f}")
        print(f"{s.id:9s}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
