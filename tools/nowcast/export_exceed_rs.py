"""Export exceedance-classifier checkpoints (train_exceed.py --export, torch .pt) to the Rust trader's JSON
(crates/processor/src/exceed.rs, kind "exceed"): member layers as dense w/b lists, mu/sd, the feature spec, the
lake column ORDER (= the bars-file column order the trainer/harness use), and torch test vectors for the Rust
parity unit test. Usage: export_exceed_rs.py OUT.json CKPT.pt [CKPT2.pt ...]"""
import glob
import json
import os
import sys

import numpy as np
import torch

out, cks = sys.argv[1], sys.argv[2:]
members, spec = [], None
for p in cks:
    ck = torch.load(p, map_location="cpu")
    assert ck.get("kind") == "exceed" and int(ck.get("nout", 1)) == 2, p
    st = ck["state"]
    layers = []
    for li in ("0", "3", "5"):
        layers.append({"w": [[round(float(v), 7) for v in row] for row in st[li + ".weight"].numpy()], "b": [round(float(v), 7) for v in st[li + ".bias"].numpy()]})
    members.append({"layers": layers})
    s = {k: ck[k] for k in ("feat", "use", "mu", "sd", "nf", "target", "x")}
    if spec is None:
        spec = s
    else:
        assert s["feat"] == spec["feat"] and list(s["use"]) == list(spec["use"]) and s["nf"] == spec["nf"] and s["target"] == spec["target"] and s["x"] == spec["x"], "member spec mismatch " + p
        assert np.allclose(s["mu"], spec["mu"]) and np.allclose(s["sd"], spec["sd"]), "member mu/sd mismatch " + p
groups = spec["feat"].split(",")
lf = [g[3:] for g in groups if g.startswith("lf_")]
bars = sorted(glob.glob("data/nowcast/lake_bars/*.bars.npz"))
assert bars, "need a lake bars file for the column order"
names = [str(x) for x in np.load(bars[-1])["names"]]
lake = [n for n in names if n in lf]                       # bars-file order (what the trainer/harness index)
assert sorted(lake) == sorted(lf), (lake, lf)
mu, sd = np.asarray(spec["mu"], np.float64), np.asarray(spec["sd"], np.float64)


def torch_score(x):
    z = torch.from_numpy(((x - mu) / sd).astype(np.float32))
    ups, dns = [], []
    for p in cks:
        ck = torch.load(p, map_location="cpu")
        net = torch.nn.Sequential(torch.nn.Linear(spec["nf"], ck["hidden"]), torch.nn.ReLU(), torch.nn.Dropout(0.0), torch.nn.Linear(ck["hidden"], ck["hidden"]), torch.nn.ReLU(), torch.nn.Linear(ck["hidden"], 2))
        net.load_state_dict(ck["state"]); net.eval()
        with torch.no_grad():
            o = torch.sigmoid(net(z)).numpy()
        ups.append(o[:, 0]); dns.append(o[:, 1])
    return np.mean(ups, 0), np.mean(dns, 0)


rng = np.random.default_rng(0)
X = mu + sd * rng.standard_normal((40, spec["nf"]))
pu, pd_ = torch_score(X)
tvs = [{"x": [round(float(v), 6) for v in X[i]], "p_up": float(pu[i]), "p_dn": float(pd_[i])} for i in range(len(X))]
js = {"kind": "exceed", "target": spec["target"], "x": float(spec["x"]),
      "note": "Exceedance classifier for the Rust trader (crates/processor/src/exceed.rs): P(mid+%s - mid >= %gc), P(<= -%gc) on the standard 43 nowcast inputs; %d members (seeds), s = P(up) - P(down). Exported from %s by tools/nowcast/export_exceed_rs.py." % (spec["target"], spec["x"], spec["x"], len(cks), ", ".join(os.path.basename(c) for c in cks)),
      "feat": spec["feat"], "lags": [int(v) for v in spec["use"]], "lake": lake, "mu": [float(v) for v in mu], "sd": [float(v) for v in sd],
      "members": members, "test_vectors": tvs}
json.dump(js, open(out, "w"), separators=(",", ":"))
print("wrote %s: nf %d, lags %s, lake %s, members %d, %.1f MB" % (out, spec["nf"], js["lags"], lake, len(members), os.path.getsize(out) / 1e6))
