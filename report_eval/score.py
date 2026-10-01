"""
Score one generated location claim against the heatmap it should describe, and
summarise many.

"Truth" here is the film's SERVED heatmap region, so these metrics measure
whether the text agrees with the evidence shown beside it, not whether it is
clinically right. The paper must say so wherever these numbers appear.
"""
from report_eval.claims import side_matches

FIELDS = ("side", "zone", "extent")


def score_output(claim, truth, text=""):
    """
    One film. `claim` is a claims.Claim or None; `truth` is the film's own
    Descriptors under the served heatmap.
    """
    row = {"claimed": claim is not None, "phantom": False, "claim_correct": False,
           "asserts_outline": "outline marks" in (text or ""),
           "outline_phantom": False}
    row["outline_phantom"] = row["asserts_outline"] and not truth.has_region
    for f in FIELDS:
        row[f"{f}_stated"] = False
        row[f"{f}_correct"] = False
    if claim is None:
        return row
    if not truth.has_region:
        row["phantom"] = True
        return row
    checks = {
        "side": claim.side is not None and side_matches(claim.side, truth),
        "zone": claim.zone is not None and claim.zone == truth.zone,
        "extent": claim.extent is not None and claim.extent == truth.extent,
    }
    for f in FIELDS:
        row[f"{f}_stated"] = getattr(claim, f) is not None
        row[f"{f}_correct"] = checks[f]
    row["claim_correct"] = all(checks[f] for f in FIELDS if row[f"{f}_stated"])
    return row


def _rate(num, den):
    return num / den if den else None


def summarise(rows):
    n = len(rows)
    claimed = [r for r in rows if r["claimed"]]
    out = {
        "n": n,
        "claims": len(claimed),
        "claim_rate": _rate(len(claimed), n),
        "correct": sum(r["claim_correct"] for r in claimed),
        "grounding_precision": _rate(sum(r["claim_correct"] for r in claimed), len(claimed)),
        "phantom_claims": sum(r["phantom"] for r in rows),
        "outline_phantoms": sum(r["outline_phantom"] for r in rows),
    }
    for f in FIELDS:
        stated = [r for r in claimed if r[f"{f}_stated"]]
        out[f"{f}_accuracy"] = _rate(sum(r[f"{f}_correct"] for r in stated), len(stated))
    return out
