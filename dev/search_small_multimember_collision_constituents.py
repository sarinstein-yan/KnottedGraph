from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import discover_yamada_theta_collisions as core
import search_prime_certified_yamada_pairs as prime_search

# Independently frozen from the audited 8-crossing discovery artifact. These are
# every collision bucket of size 3 or 4 (96 diagrams total). Bucket membership
# is used only for candidate discovery; any selected pair is re-certified from
# scratch for equal Yamada before a theorem claim.
BUCKETS = [(45, [[9,23,.05],[9,113,.05],[10,123,.12],[13,139,.08]]),(46, [[9,142,.05],[9,232,.05],[10,132,.12],[13,116,.08]]),(47, [[10,84,.12],[16,231,.12],[16,246,.12],[20,99,.12]]),(48, [[10,116,.12],[20,134,.12],[23,139,.12],[23,155,.12]]),(49, [[10,139,.12],[20,121,.12],[23,100,.12],[23,116,.12]]),(50, [[10,171,.12],[16,9,.12],[16,24,.12],[20,156,.12]]),(51, [[20,70,.12],[23,211,.12],[33,30,.05],[33,86,.05]]),(52, [[20,84,.12],[23,243,.12],[24,132,.12],[24,232,.12]]),(53, [[20,171,.12],[23,12,.12],[24,23,.12],[24,123,.12]]),(54, [[20,185,.12],[23,44,.12],[33,169,.05],[33,225,.05]]),(55, [[24,20,.12],[24,235,.12],[25,84,.12],[25,171,.12]]),(56, [[24,88,.12],[24,143,.12],[25,54,.12],[25,237,.12]]),(57, [[24,112,.12],[24,167,.12],[25,18,.12],[25,201,.12]]),(58, [[32,56,.12],[32,122,.12],[39,145,.05],[39,157,.05]]),(59, [[32,133,.12],[32,199,.12],[39,98,.05],[39,110,.05]]),(60, [[7,35,.12],[15,67,.12],[15,73,.12]]),(61, [[7,220,.12],[15,182,.12],[15,188,.12]]),(62, [[9,67,.05],[9,91,.05],[23,39,.12]]),(63, [[9,164,.05],[9,188,.05],[23,216,.12]]),(64, [[10,92,.12],[20,107,.12],[20,150,.12]]),(65, [[10,163,.12],[20,105,.12],[20,148,.12]]),(66, [[13,83,.08],[23,43,.12],[33,81,.05]]),(67, [[13,172,.08],[23,212,.12],[33,174,.05]]),(68, [[16,19,.12],[32,19,.12],[39,248,.05]]),(69, [[16,236,.12],[32,236,.12],[39,7,.05]]),(70, [[24,4,.12],[24,233,.12],[33,156,.05]]),(71, [[24,22,.12],[24,251,.12],[33,99,.05]])]


def classify(shadows, desc):
    shadow,bits,fraction=desc
    _,raw=core.spatial_theta(shadows[shadow],bits,approach_fraction=fraction)
    edges=[np.asarray(p,dtype=float) for p in raw]
    ordered=prime_search.ordered_constituents(edges)
    values=list(ordered.values())
    resolved=all("TMC" not in value for value in values)
    return {"shadow":shadow,"bits":bits,"approach_fraction":fraction,
            "constituents_ordered":ordered,
            "constituent_multiset":sorted(values) if resolved else None,
            "fully_resolved":resolved,"has_unknot":"0_1" in values}


def run(plantri: str, output: Path):
    shadows={s.index:s for s in core.generate_shadows(plantri,8)}
    cache={}; promising=[]
    for bucket_index,members in BUCKETS:
        groups={}; unresolved=[]
        for desc in members:
            key=(desc[0],desc[1])
            if key not in cache: cache[key]=classify(shadows,desc)
            member=cache[key]
            if not member["fully_resolved"]:
                unresolved.append(member); continue
            groups.setdefault(tuple(member["constituent_multiset"]),[]).append(member)
        if len(groups)<2: continue
        signature_groups=[]
        for sig,group in sorted(groups.items(),key=lambda kv:repr(kv[0])):
            signature_groups.append({"signature":list(sig),"members":group,
                                     "has_unknot_member":any(x["has_unknot"] for x in group)})
        record={"bucket_index":bucket_index,"bucket_size":len(members),
                "signature_groups":signature_groups,"unresolved_members":unresolved,
                "two_unknot_signature_groups_available":sum(g["has_unknot_member"] for g in signature_groups)>=2}
        promising.append(record)
        print("SMALL_MULTIMEMBER_BREAKTHROUGH="+json.dumps(record,sort_keys=True),flush=True)
    promising.sort(key=lambda r:(not r["two_unknot_signature_groups_available"],r["bucket_size"],r["bucket_index"]))
    payload={"scanned_bucket_count":len(BUCKETS),"scanned_member_count":sum(len(m) for _,m in BUCKETS),
             "constituent_distinct_bucket_count":len(promising),
             "prime_lift_ready_bucket_count":sum(r["two_unknot_signature_groups_available"] for r in promising),
             "promising_buckets":promising}
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,indent=2,sort_keys=True))
    print("SMALL_MULTIMEMBER_SUMMARY="+json.dumps({k:payload[k] for k in ("scanned_bucket_count","scanned_member_count","constituent_distinct_bucket_count","prime_lift_ready_bucket_count")},sort_keys=True),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--plantri",required=True); ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args(); run(args.plantri,args.output)

if __name__=="__main__": main()
