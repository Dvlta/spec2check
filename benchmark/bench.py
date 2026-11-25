"""Spec2Check: natural-language porosity acceptance specs -> executable checks.

Each spec has one reference implementation per reasonable interpretation (ambiguous specs have >= 2).
A part is a pore table as exported by CT porosity analysis; check(part) returns True to accept.
"""
import itertools
import math
import random
from dataclasses import dataclass, field

SCHEMA = '''part = {
  "part_id": str,
  "part_volume_mm3": float,      # volume enclosed by the part's outer surface (pores included)
  "nominal_wall_mm": float,      # nominal wall thickness from the drawing
  "zones": [                     # named inspection zones, axis-aligned boxes in mm; zones may overlap
    {"name": str, "box": [xmin, xmax, ymin, ymax, zmin, zmax]}
  ],
  "pores": [
    {
      "id": int,
      "centroid": [x, y, z],                         # mm
      "bbox": [xmin, xmax, ymin, ymax, zmin, zmax],  # mm, axis-aligned bounding box of the pore
      "volume_mm3": float,
      "d_eq_mm": float,          # diameter of the sphere with the same volume as the pore
      "feret_max_mm": float,     # largest caliper (Feret) diameter of the pore
      "sphericity": float,       # 1.0 = sphere; shrinkage pores are typically below 0.5
      "dist_surface_mm": float,  # shortest distance from the pore boundary to the outer surface (0 = open to surface)
      "wall_mm": float           # local wall thickness at the pore location
    }
  ]
}'''


@dataclass
class Spec:
    id: str
    tier: str            # T1 single criterion, T2 zones/conditions, T3 aggregate/spatial, T4 ambiguous
    text: str
    refs: dict           # interpretation name -> function(part) -> bool (True = accept)
    ambiguity: str = ""  # what is ambiguous (T4 only)
    snap: dict = field(default_factory=dict)  # pore attribute -> thresholds to probe exactly


# ---------- helpers ----------
def in_box(p, box):
    x, y, z = p
    return box[0] <= x <= box[1] and box[2] <= y <= box[3] and box[4] <= z <= box[5]


def box_overlap(a, b):
    return a[0] <= b[1] and b[0] <= a[1] and a[2] <= b[3] and b[2] <= a[3] and a[4] <= b[5] and b[4] <= a[5]


def zone(part, name):
    return [z["box"] for z in part["zones"] if z["name"] == name]


def centroid_in(part, pore, name):
    return any(in_box(pore["centroid"], b) for b in zone(part, name))


def bbox_in(part, pore, name):
    return any(box_overlap(pore["bbox"], b) for b in zone(part, name))


def dist(a, b):
    return math.dist(a["centroid"], b["centroid"])


def box_volume(b):
    return (b[1] - b[0]) * (b[3] - b[2]) * (b[5] - b[4])


def max_window_sum(items, edge, axes=(0, 1, 2)):
    """Largest summed weight of points inside any axis-aligned window of the given edge (faces inclusive), any position.

    items: list of (point, weight); axes: which coordinates the window spans (2 axes = a projected area).
    """
    if not items:
        return 0.0
    best = 0.0

    def rec(cands, k):
        nonlocal best
        if k == len(axes):
            best = max(best, sum(w for _, w in cands))
            return
        ax = axes[k]
        for lo in sorted({p[ax] for p, _ in cands}):
            sub = [(p, w) for p, w in cands if lo <= p[ax] <= lo + edge]
            if sum(w for _, w in sub) > best:
                rec(sub, k + 1)
    rec(items, 0)
    return best


def aligned_length(pores, tol, size_key):
    """Largest summed size of pores lying within tol of a common line through a pore, parallel to an axis."""
    best = 0.0
    for p in pores:
        for ax in range(3):
            others = [o for o in range(3) if o != ax]
            grp = [q for q in pores if all(abs(q["centroid"][o] - p["centroid"][o]) <= tol for o in others)]
            best = max(best, sum(q[size_key] for q in grp))
    return best


def max_in_window(points, edge):
    """Max number of points inside any axis-aligned cube of the given edge (faces inclusive), any position."""
    best = 0
    xs = sorted({p[0] for p in points}); ys = sorted({p[1] for p in points}); zs = sorted({p[2] for p in points})
    for x0 in xs:
        sx = [p for p in points if x0 <= p[0] <= x0 + edge]
        if len(sx) <= best:
            continue
        for y0 in ys:
            sy = [p for p in sx if y0 <= p[1] <= y0 + edge]
            if len(sy) <= best:
                continue
            for z0 in zs:
                n = sum(1 for p in sy if z0 <= p[2] <= z0 + edge)
                best = max(best, n)
    return best


def max_in_grid(points, edge):
    """Max number of points in one cell of a fixed grid of cubes anchored at the origin."""
    cells = {}
    for p in points:
        key = tuple(math.floor(c / edge) for c in p)
        cells[key] = cells.get(key, 0) + 1
    return max(cells.values(), default=0)


def clusters(pores, neighbor):
    parent = list(range(len(pores)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, j in itertools.combinations(range(len(pores)), 2):
        if neighbor(pores[i], pores[j]):
            parent[find(i)] = find(j)
    groups = {}
    for i in range(len(pores)):
        groups.setdefault(find(i), []).append(pores[i])
    return list(groups.values())


def nearest_rank(values, q):
    v = sorted(values)
    return v[max(0, math.ceil(q * len(v)) - 1)]


def round_half_up(x, step):
    return math.floor(x / step + 0.5 + 1e-12) * step


P = lambda part: part["pores"]  # noqa: E731


# ---------- specs ----------
def specs():
    S = []
    add = lambda *a, **k: S.append(Spec(*a, **k))  # noqa: E731

    # T1: single criterion, fully specified
    add("T1-01", "T1", "Reject the part if any pore has an equivalent diameter greater than 0.50 mm.",
        {"ref": lambda t: all(p["d_eq_mm"] <= 0.50 for p in P(t))}, snap={"d_eq_mm": [0.50]})
    add("T1-02", "T1", "The total pore volume must not exceed 0.30% of the part volume (part_volume_mm3). Reject the part otherwise.",
        {"ref": lambda t: sum(p["volume_mm3"] for p in P(t)) <= 0.003 * t["part_volume_mm3"]})
    add("T1-03", "T1", "Reject the part if it contains more than 25 pores with an equivalent diameter of 0.10 mm or larger.",
        {"ref": lambda t: sum(p["d_eq_mm"] >= 0.10 for p in P(t)) <= 25}, snap={"d_eq_mm": [0.10]})
    add("T1-04", "T1", "Reject the part if any pore with sphericity below 0.40 has a maximum Feret diameter larger than 0.30 mm.",
        {"ref": lambda t: not any(p["sphericity"] < 0.40 and p["feret_max_mm"] > 0.30 for p in P(t))},
        snap={"sphericity": [0.40], "feret_max_mm": [0.30]})
    add("T1-05", "T1", "Reject the part if any pore with an equivalent diameter of at least 0.15 mm lies less than 0.50 mm from the outer surface (dist_surface_mm).",
        {"ref": lambda t: not any(p["d_eq_mm"] >= 0.15 and p["dist_surface_mm"] < 0.50 for p in P(t))},
        snap={"d_eq_mm": [0.15], "dist_surface_mm": [0.50]})
    add("T1-06", "T1", "Reject the part if any pore's maximum Feret diameter exceeds 20% of the local wall thickness at that pore.",
        {"ref": lambda t: all(p["feret_max_mm"] <= 0.20 * p["wall_mm"] for p in P(t))})
    add("T1-07", "T1", "The summed volume of all pores with an equivalent diameter above 0.20 mm must not exceed 0.050 mm^3. Reject the part otherwise.",
        {"ref": lambda t: sum(p["volume_mm3"] for p in P(t) if p["d_eq_mm"] > 0.20) <= 0.050}, snap={"d_eq_mm": [0.20]})
    add("T1-08", "T1", "Reject the part if the largest pore by volume has a sphericity below 0.30. A part without pores is accepted.",
        {"ref": lambda t: not P(t) or max(P(t), key=lambda p: p["volume_mm3"])["sphericity"] >= 0.30})

    # T2: zones and conditional limits, fully specified
    add("T2-01", "T2", "A pore whose centroid lies inside the zone named 'sealing' (boundary inclusive) must have an equivalent diameter of at most 0.20 mm; every other pore must have an equivalent diameter of at most 0.80 mm. Reject the part if any pore exceeds the limit that applies to it. A part without a zone of that name has no pores in it.",
        {"ref": lambda t: all(p["d_eq_mm"] <= (0.20 if centroid_in(t, p, "sealing") else 0.80) for p in P(t))},
        snap={"d_eq_mm": [0.20, 0.80]})
    add("T2-02", "T2", "Inside zone 'bolt_boss' (pore centroid inside, boundary inclusive), no pore with an equivalent diameter of 0.10 mm or more may lie less than 1.0 mm from the outer surface. Outside that zone the same restriction applies only to pores with an equivalent diameter of 0.30 mm or more. A part without a zone of that name has no pores in it.",
        {"ref": lambda t: not any(p["dist_surface_mm"] < 1.0 and p["d_eq_mm"] >= (0.10 if centroid_in(t, p, "bolt_boss") else 0.30) for p in P(t))},
        snap={"d_eq_mm": [0.10, 0.30], "dist_surface_mm": [1.0]})
    add("T2-03", "T2", "The allowable equivalent diameter of a pore is 25% of the local wall thickness at the pore, but never more than 1.5 mm. Reject the part if any pore exceeds its allowable size.",
        {"ref": lambda t: all(p["d_eq_mm"] <= min(0.25 * p["wall_mm"], 1.5) for p in P(t))})
    add("T2-04", "T2", "Zones are ranked sealing > machined > everything else. Each pore is assigned to the highest-ranked zone that contains its centroid (boundaries inclusive). Equivalent-diameter limits: sealing 0.20 mm, machined 0.40 mm, everything else 1.00 mm. Reject the part if any pore exceeds the limit of its assigned zone. A part may lack any of these zones.",
        {"ref": lambda t: all(p["d_eq_mm"] <= (0.20 if centroid_in(t, p, "sealing") else 0.40 if centroid_in(t, p, "machined") else 1.00) for p in P(t))},
        snap={"d_eq_mm": [0.20, 0.40]})
    add("T2-05", "T2", "At most 3 pores with an equivalent diameter of 0.15 mm or more may have their centroid in zone 'machined' (boundary inclusive). Reject the part otherwise. A part without a zone of that name has no pores in it.",
        {"ref": lambda t: sum(p["d_eq_mm"] >= 0.15 and centroid_in(t, p, "machined") for p in P(t)) <= 3}, snap={"d_eq_mm": [0.15]})
    add("T2-06", "T2", "Classify each pore as a gas pore if its sphericity is 0.60 or higher, otherwise as a shrinkage pore. Gas pores are allowed up to an equivalent diameter of 0.60 mm; shrinkage pores are allowed up to a maximum Feret diameter of 0.40 mm. In addition, at most 10 shrinkage pores (of any size) are allowed. Reject the part if any of these limits is exceeded.",
        {"ref": lambda t: all((p["d_eq_mm"] <= 0.60) if p["sphericity"] >= 0.60 else (p["feret_max_mm"] <= 0.40) for p in P(t))
         and sum(p["sphericity"] < 0.60 for p in P(t)) <= 10},
        snap={"sphericity": [0.60], "d_eq_mm": [0.60], "feret_max_mm": [0.40]})
    add("T2-07", "T2", "Within each zone named 'sealing', the summed volume of the pores whose centroids lie in that zone box (boundary inclusive) must not exceed 0.050% of the zone box volume. Reject the part otherwise. A part without a zone of that name has no pores in it.",
        {"ref": lambda t: all(sum(p["volume_mm3"] for p in P(t) if in_box(p["centroid"], b)) <= 0.0005 * box_volume(b) for b in zone(t, "sealing"))})
    add("T2-08", "T2", "A pore that is open to the surface (dist_surface_mm equal to 0) is rejectable regardless of size if its centroid lies in zone 'sealing' or zone 'machined' (boundary inclusive); anywhere else it is rejectable only if its equivalent diameter exceeds 0.30 mm. Pores not open to the surface are not covered by this rule. A part without a zone of that name has no pores in it.",
        {"ref": lambda t: not any(p["dist_surface_mm"] == 0 and (centroid_in(t, p, "sealing") or centroid_in(t, p, "machined") or p["d_eq_mm"] > 0.30) for p in P(t))},
        snap={"d_eq_mm": [0.30], "dist_surface_mm": [0.0]})

    # T3: aggregate / spatial, fully specified but algorithmically harder
    big = lambda t, a: [p for p in P(t) if p["d_eq_mm"] >= a]  # noqa: E731
    add("T3-01", "T3", "Consider every axis-aligned cube with an edge of 5.0 mm, at any position. No such cube may contain the centroids of more than 4 pores with an equivalent diameter of 0.10 mm or more (a centroid on a cube face counts as inside). Reject the part otherwise.",
        {"ref": lambda t: max_in_window([p["centroid"] for p in big(t, 0.10)], 5.0) <= 4})
    add("T3-02", "T3", "Any two pores that both have an equivalent diameter of 0.20 mm or more must be separated by a centroid-to-centroid distance of at least 3 times the larger of their two equivalent diameters. Reject the part otherwise.",
        {"ref": lambda t: all(dist(a, b) >= 3 * max(a["d_eq_mm"], b["d_eq_mm"]) for a, b in itertools.combinations(big(t, 0.20), 2))})
    add("T3-03", "T3", "Only pores with an equivalent diameter of 0.05 mm or more are considered. Two such pores are neighbours if the gap between them (centroid distance minus half the sum of their equivalent diameters) is less than 0.50 mm. A cluster is a group of pores connected through chains of neighbours; a pore without neighbours is a cluster of one. Reject the part if any cluster contains more than 5 pores or if the summed equivalent diameter of any cluster exceeds 1.0 mm.",
        {"ref": lambda t: all(len(c) <= 5 and sum(p["d_eq_mm"] for p in c) <= 1.0 for c in clusters(big(t, 0.05), lambda a, b: dist(a, b) - (a["d_eq_mm"] + b["d_eq_mm"]) / 2 < 0.50))})
    add("T3-04", "T3", "Project the centroids of all pores with an equivalent diameter of 0.10 mm or more onto the x axis. Within any interval of length 10.0 mm along x (any position, ends inclusive), the summed equivalent diameter of the pores in the interval must not exceed 1.2 mm. Reject the part otherwise.",
        {"ref": lambda t: all(sum(p["d_eq_mm"] for p in big(t, 0.10) if a["centroid"][0] <= p["centroid"][0] <= a["centroid"][0] + 10.0) <= 1.2 for a in big(t, 0.10))})
    add("T3-05", "T3", "Using the nearest-rank method, the 95th percentile of the equivalent diameters of all pores must not exceed 0.30 mm. A part without pores is accepted. Reject the part otherwise.",
        {"ref": lambda t: not P(t) or nearest_rank([p["d_eq_mm"] for p in P(t)], 0.95) <= 0.30})
    add("T3-06", "T3", "The pore density, defined as the number of pores with an equivalent diameter of 0.05 mm or more per cubic centimetre of part volume, must not exceed 40. Reject the part otherwise.",
        {"ref": lambda t: len(big(t, 0.05)) / (t["part_volume_mm3"] / 1000.0) <= 40})
    add("T3-07", "T3", "A pore counts as being in a zone if its bounding box touches or overlaps the zone box. Every pore in zone 'sealing' must have a maximum Feret diameter of at most 0.25 mm. Reject the part otherwise. A part without a zone of that name has no pores in it.",
        {"ref": lambda t: all(p["feret_max_mm"] <= 0.25 for p in P(t) if bbox_in(t, p, "sealing"))}, snap={"feret_max_mm": [0.25]})
    add("T3-08", "T3", "For every zone, compute its porosity as the summed volume of the pores whose centroid lies in the zone box (boundary inclusive) divided by the zone box volume; a pore inside several overlapping zones counts for each of them. Reject the part if any zone has a porosity above 0.10%.",
        {"ref": lambda t: all(sum(p["volume_mm3"] for p in P(t) if in_box(p["centroid"], z["box"])) <= 0.001 * box_volume(z["box"]) for z in t["zones"])})

    # T4: ambiguous or underspecified; one reference per reasonable reading
    add("T4-01", "T4", "Reject the part if any pore is larger than 0.40 mm.",
        {"d_eq": lambda t: all(p["d_eq_mm"] <= 0.40 for p in P(t)),
         "feret_max": lambda t: all(p["feret_max_mm"] <= 0.40 for p in P(t))},
        ambiguity="pore size measure (equivalent diameter vs maximum Feret)")
    add("T4-02", "T4", "No pore with an equivalent diameter above 0.20 mm is permitted in the sealing zone. A part without a zone of that name has no pores in it.",
        {"centroid": lambda t: not any(p["d_eq_mm"] > 0.20 and centroid_in(t, p, "sealing") for p in P(t)),
         "any_overlap": lambda t: not any(p["d_eq_mm"] > 0.20 and bbox_in(t, p, "sealing") for p in P(t))},
        ambiguity="zone membership (centroid vs any part of the pore)")
    add("T4-03", "T4", "At most 3 pores with an equivalent diameter of 0.15 mm or more are permitted per 5 mm x 5 mm x 5 mm reference volume.",
        {"sliding": lambda t: max_in_window([p["centroid"] for p in big(t, 0.15)], 5.0) <= 3,
         "fixed_grid": lambda t: max_in_grid([p["centroid"] for p in big(t, 0.15)], 5.0) <= 3},
        ambiguity="reference volume placement (any position vs fixed grid)")
    add("T4-04", "T4", "Pores with an equivalent diameter of 0.20 mm or more must be at least 1.0 mm apart.",
        {"center": lambda t: all(dist(a, b) >= 1.0 for a, b in itertools.combinations(big(t, 0.20), 2)),
         "edge": lambda t: all(dist(a, b) - (a["d_eq_mm"] + b["d_eq_mm"]) / 2 >= 1.0 for a, b in itertools.combinations(big(t, 0.20), 2))},
        ambiguity="spacing measure (centroid-to-centroid vs edge-to-edge)")
    add("T4-05", "T4", "Porosity must not exceed 0.5%.",
        {"global": lambda t: sum(p["volume_mm3"] for p in P(t)) <= 0.005 * t["part_volume_mm3"],
         "per_zone": lambda t: all(sum(p["volume_mm3"] for p in P(t) if in_box(p["centroid"], z["box"])) <= 0.005 * box_volume(z["box"]) for z in t["zones"])
         and sum(p["volume_mm3"] for p in P(t)) <= 0.005 * t["part_volume_mm3"]},
        ambiguity="porosity scope (whole part vs also every local zone)")
    add("T4-06", "T4", "Pores near the surface must not exceed an equivalent diameter of 0.15 mm.",
        {"near_0.5": lambda t: not any(p["dist_surface_mm"] < 0.5 and p["d_eq_mm"] > 0.15 for p in P(t)),
         "near_1.0": lambda t: not any(p["dist_surface_mm"] < 1.0 and p["d_eq_mm"] > 0.15 for p in P(t)),
         "near_2.0": lambda t: not any(p["dist_surface_mm"] < 2.0 and p["d_eq_mm"] > 0.15 for p in P(t))},
        ambiguity="undefined 'near the surface' distance")
    add("T4-07", "T4", "Pore sizes (equivalent diameter) are reported to 0.1 mm. A pore of 0.5 mm or less is acceptable; reject the part if any pore is larger.",
        {"absolute": lambda t: all(p["d_eq_mm"] <= 0.5 for p in P(t)),
         "rounded": lambda t: all(round_half_up(p["d_eq_mm"], 0.1) <= 0.5 + 1e-9 for p in P(t))},
        ambiguity="compare raw measurement vs value rounded to the reporting precision")
    add("T4-08", "T4", "Clustered porosity is not permitted.",
        {"3_within_1mm": lambda t: not any(sum(dist(a, b) <= 1.0 for b in big(t, 0.10)) >= 3 for a in big(t, 0.10)),
         "gap_lt_size": lambda t: all(len(c) < 3 for c in clusters(big(t, 0.10), lambda a, b: dist(a, b) - (a["d_eq_mm"] + b["d_eq_mm"]) / 2 < max(a["d_eq_mm"], b["d_eq_mm"]))),
         "4_within_2mm": lambda t: not any(sum(dist(a, b) <= 2.0 for b in big(t, 0.05)) >= 4 for a in big(t, 0.05))},
        ambiguity="undefined cluster definition")
    add("T4-09", "T4", "Pores (equivalent diameter) must not exceed 25% of the wall thickness.",
        {"local": lambda t: all(p["d_eq_mm"] <= 0.25 * p["wall_mm"] for p in P(t)),
         "nominal": lambda t: all(p["d_eq_mm"] <= 0.25 * t["nominal_wall_mm"] for p in P(t))},
        ambiguity="local wall thickness at the pore vs nominal wall thickness")

    # T4 (extended): ambiguities modelled on recurring issues in inspection standards practice
    vol = lambda t, ps: sum(p["volume_mm3"] for p in ps)  # noqa: E731
    in_z = lambda t, name: [p for p in P(t) if centroid_in(t, p, name)]  # noqa: E731
    add("T4-10", "T4", "No more than 15 pores are permitted. Pores below the detection limit of the CT system are not counted.",
        {"count_all": lambda t: len(P(t)) <= 15,
         "limit_0.05": lambda t: sum(p["d_eq_mm"] >= 0.05 for p in P(t)) <= 15,
         "limit_0.10": lambda t: sum(p["d_eq_mm"] >= 0.10 for p in P(t)) <= 15},
        ambiguity="undefined detection limit (every listed pore is already a detection vs a size cut-off)")
    add("T4-11", "T4", "The total length of aligned pores must not exceed 2.0 mm.",
        {"tol0.5_feret": lambda t: aligned_length(P(t), 0.5, "feret_max_mm") <= 2.0,
         "tol0.5_deq": lambda t: aligned_length(P(t), 0.5, "d_eq_mm") <= 2.0,
         "tol0.2_feret": lambda t: aligned_length(P(t), 0.2, "feret_max_mm") <= 2.0},
        ambiguity="undefined alignment tolerance and pore length measure")
    add("T4-12", "T4", "Pores with an equivalent diameter above 0.30 mm are not permitted at sealing surfaces. A part without a zone named 'sealing' has no sealing surfaces.",
        {"whole_zone": lambda t: not any(p["d_eq_mm"] > 0.30 for p in in_z(t, "sealing")),
         "open_to_surface": lambda t: not any(p["d_eq_mm"] > 0.30 and p["dist_surface_mm"] == 0 for p in in_z(t, "sealing")),
         "within_0.5": lambda t: not any(p["d_eq_mm"] > 0.30 and p["dist_surface_mm"] < 0.5 for p in in_z(t, "sealing"))},
        ambiguity="whether 'at sealing surfaces' means anywhere in the sealing zone, open to the surface, or near it")
    add("T4-13", "T4", "Shrinkage porosity is not permitted. Gas porosity is acceptable up to an equivalent diameter of 0.50 mm.",
        {"shrink<0.5": lambda t: all(p["sphericity"] >= 0.5 and p["d_eq_mm"] <= 0.5 for p in P(t)),
         "shrink<0.6": lambda t: all(p["sphericity"] >= 0.6 and p["d_eq_mm"] <= 0.5 for p in P(t)),
         "shrink<0.7": lambda t: all(p["sphericity"] >= 0.7 and p["d_eq_mm"] <= 0.5 for p in P(t))},
        ambiguity="undefined sphericity boundary between shrinkage and gas pores")
    add("T4-14", "T4", "Pores closer than 0.5 mm to the surface must not exceed an equivalent diameter of 0.15 mm.",
        {"boundary": lambda t: not any(p["dist_surface_mm"] < 0.5 and p["d_eq_mm"] > 0.15 for p in P(t)),
         "centroid": lambda t: not any(p["dist_surface_mm"] + p["d_eq_mm"] / 2 < 0.5 and p["d_eq_mm"] > 0.15 for p in P(t))},
        ambiguity="distance to the surface measured from the pore boundary or from its centre")
    add("T4-15", "T4", "Porosity must not exceed 1% in any 10 mm x 10 mm reference area.",
        {"projected_area": lambda t: max_window_sum([(p["centroid"], math.pi * p["d_eq_mm"] ** 2 / 4) for p in P(t)], 10.0, (0, 1)) <= 0.01 * 100,
         "cube_volume": lambda t: max_window_sum([(p["centroid"], p["volume_mm3"]) for p in P(t)], 10.0) <= 0.01 * 1000},
        ambiguity="a 2D reference area (projected pore area) vs a 3D reference volume in CT data")
    add("T4-16", "T4", "At most 2 pores with an equivalent diameter of 0.15 mm or more are permitted per machined zone (pore centroid inside, boundary inclusive). A part without a zone named 'machined' has no pores in it.",
        {"per_box": lambda t: all(sum(p["d_eq_mm"] >= 0.15 and in_box(p["centroid"], b) for p in P(t)) <= 2 for b in zone(t, "machined")),
         "per_name": lambda t: sum(p["d_eq_mm"] >= 0.15 for p in in_z(t, "machined")) <= 2},
        ambiguity="each 'machined' box separately vs all boxes named 'machined' together")
    add("T4-17", "T4", "Pores with an equivalent diameter of 0.20 mm or more must be separated, centroid to centroid, by at least twice the pore diameter.",
        {"larger": lambda t: all(dist(a, b) >= 2 * max(a["d_eq_mm"], b["d_eq_mm"]) for a, b in itertools.combinations(big(t, 0.20), 2)),
         "smaller": lambda t: all(dist(a, b) >= 2 * min(a["d_eq_mm"], b["d_eq_mm"]) for a, b in itertools.combinations(big(t, 0.20), 2)),
         "mean": lambda t: all(dist(a, b) >= a["d_eq_mm"] + b["d_eq_mm"] for a, b in itertools.combinations(big(t, 0.20), 2))},
        ambiguity="which pore's diameter sets the required spacing (larger, smaller or mean)")
    add("T4-18", "T4", "Pores with an equivalent diameter larger than 0.30 mm are rejectable. The pore-size measurement uncertainty of the CT system is 0.05 mm.",
        {"ignore_U": lambda t: all(p["d_eq_mm"] <= 0.30 for p in P(t)),
         "prove_conformance": lambda t: all(p["d_eq_mm"] + 0.05 <= 0.30 for p in P(t)),
         "prove_nonconformance": lambda t: all(p["d_eq_mm"] - 0.05 <= 0.30 for p in P(t))},
        ambiguity="decision rule for measurement uncertainty (ignore it, require proof of conformance, or require proof of nonconformance)")
    add("T4-19", "T4", "The average pore size must not exceed 0.15 mm. A part without pores is accepted.",
        {"mean_deq": lambda t: not P(t) or sum(p["d_eq_mm"] for p in P(t)) / len(P(t)) <= 0.15,
         "volume_weighted": lambda t: not P(t) or sum(p["d_eq_mm"] * p["volume_mm3"] for p in P(t)) / sum(p["volume_mm3"] for p in P(t)) <= 0.15,
         "mean_feret": lambda t: not P(t) or sum(p["feret_max_mm"] for p in P(t)) / len(P(t)) <= 0.15},
        ambiguity="which average (count-weighted or volume-weighted) and which size measure")
    add("T4-20", "T4", "Pore density must not exceed 10 pores per cubic centimetre.",
        {"global": lambda t: len(P(t)) / (t["part_volume_mm3"] / 1000.0) <= 10,
         "local_cube": lambda t: max_window_sum([(p["centroid"], 1.0) for p in P(t)], 10.0) <= 10},
        ambiguity="density over the whole part vs in any 1 cm cube")
    add("T4-21", "T4", "Porosity in the sealing zone must not exceed 0.2%. A part without a zone named 'sealing' has no pores in it.",
        {"zone_volume": lambda t: not zone(t, "sealing") or vol(t, in_z(t, "sealing")) <= 0.002 * sum(box_volume(b) for b in zone(t, "sealing")),
         "part_volume": lambda t: vol(t, in_z(t, "sealing")) <= 0.002 * t["part_volume_mm3"]},
        ambiguity="porosity of the zone relative to the zone volume or to the whole part volume")
    return S


# ---------- part generation ----------
ZONE_NAMES = ["sealing", "machined", "bolt_boss"]


def make_pore(rng, i, center, d_eq, sph, part_dims):
    d_eq = max(d_eq, 0.01)
    feret = d_eq / max(sph, 0.2) ** 0.8 * rng.uniform(1.0, 1.15)
    half = [feret / 2 * rng.uniform(0.6, 1.0) for _ in range(3)]
    half[rng.randrange(3)] = feret / 2
    x, y, z = center
    surf = min(x, part_dims[0] - x, y, part_dims[1] - y, z, part_dims[2] - z) - feret / 2
    return {
        "id": i, "centroid": [round(c, 4) for c in center],
        "bbox": [round(x - half[0], 4), round(x + half[0], 4), round(y - half[1], 4), round(y + half[1], 4),
                 round(z - half[2], 4), round(z + half[2], 4)],
        "volume_mm3": round(math.pi / 6 * d_eq ** 3, 7), "d_eq_mm": round(d_eq, 4), "feret_max_mm": round(feret, 4),
        "sphericity": round(sph, 3), "dist_surface_mm": round(max(0.0, surf), 4),
        "wall_mm": round(rng.uniform(1.5, 8.0), 3),
    }


def make_part(rng, pid, n_mean=12, size_scale=0.12, cluster_p=0.3, scale=1.0):
    dims = (rng.uniform(30, 60) * scale, rng.uniform(20, 40) * scale, rng.uniform(10, 25) * scale)
    zones = []
    for name in ZONE_NAMES:
        for _ in range(rng.choice([0, 1, 1, 2])):
            w = [rng.uniform(3, 12) * scale, rng.uniform(3, 12) * scale, rng.uniform(2, 8) * scale]
            o = [rng.uniform(0, dims[k] - w[k]) for k in range(3)]
            zones.append({"name": name, "box": [round(v, 3) for k in range(3) for v in (o[k], o[k] + w[k])]})
    n = max(0, int(rng.expovariate(1 / n_mean)))
    pores = []
    centers = []
    for i in range(n):
        if centers and rng.random() < cluster_p:
            c0 = rng.choice(centers)
            c = [min(max(c0[k] + rng.gauss(0, 0.8 * min(scale, 1.0)), 0.05), dims[k] - 0.05) for k in range(3)]
        elif zones and rng.random() < 0.35:
            b = rng.choice(zones)["box"]
            c = [rng.uniform(b[2 * k] - 0.4, b[2 * k + 1] + 0.4) for k in range(3)]
            c = [min(max(c[k], 0.05), dims[k] - 0.05) for k in range(3)]
        else:
            c = [rng.uniform(0.05, dims[k] - 0.05) for k in range(3)]
        centers.append(c)
        d = rng.lognormvariate(math.log(size_scale), 0.6)
        sph = rng.betavariate(5, 2) if rng.random() < 0.7 else rng.betavariate(2, 4)
        pores.append(make_pore(rng, i, c, d, sph, dims))
    vol = dims[0] * dims[1] * dims[2] * rng.uniform(0.25, 0.6)
    return {"part_id": pid, "part_volume_mm3": round(vol, 2), "nominal_wall_mm": round(rng.uniform(2.0, 6.0), 2),
            "zones": zones, "pores": pores}


def snapped(rng, part, attr, value):
    """Copy of part with one pore's attribute set exactly to a threshold (boundary test)."""
    if not part["pores"]:
        return None
    q = {**part, "pores": [dict(p) for p in part["pores"]]}
    p = rng.choice(q["pores"])
    p[attr] = value
    if attr == "d_eq_mm":
        p["volume_mm3"] = round(math.pi / 6 * value ** 3, 7)
        p["feret_max_mm"] = max(p["feret_max_mm"], value)
    if attr == "feret_max_mm":
        p["d_eq_mm"] = min(p["d_eq_mm"], value)
        p["volume_mm3"] = round(math.pi / 6 * p["d_eq_mm"] ** 3, 7)
    q["part_id"] += f"-snap-{attr}"
    return q


def test_parts(spec, seed=0, n_random=400, n_snap=120, n_disc=60):
    """Random parts, boundary (snapped) parts, and for ambiguous specs parts where readings disagree."""
    rng = random.Random(hash(spec.id) % 10_000 + seed)
    parts = []
    for i in range(n_random):
        parts.append(make_part(rng, f"{spec.id}-r{i}", n_mean=rng.choice([4, 12, 30]),
                               size_scale=rng.choice([0.06, 0.12, 0.25]), cluster_p=rng.choice([0.0, 0.3, 0.7]),
                               scale=rng.choice([1.0, 1.0, 0.3, 0.12])))
    for i in range(n_snap if spec.snap else 0):
        attr = rng.choice(list(spec.snap))
        q = snapped(rng, make_part(rng, f"{spec.id}-s{i}"), attr, rng.choice(spec.snap[attr]))
        if q:
            parts.append(q)
    disc = []
    if len(spec.refs) > 1:
        tries = 0
        while len(disc) < n_disc and tries < 20000:
            tries += 1
            q = make_part(rng, f"{spec.id}-d{tries}", n_mean=rng.choice([6, 15, 30]),
                          size_scale=rng.choice([0.1, 0.2, 0.3]), cluster_p=rng.choice([0.3, 0.7]))
            if len({f(q) for f in spec.refs.values()}) > 1:
                disc.append(q)
    return parts, disc
