"""One trial through the HSMM detectors, tick by tick: the readable entry point (docs/README.md).

Loads a model, takes one test trial (optionally with an injected error), streams it through
Live and Lag(5) one tick at a time, then runs Hindsight on the whole trial and narrates it.

    ./py scripts/demo_trial.py                      # first Breakfast test trial, healthy
    ./py scripts/demo_trial.py --trial 3 --inject omission
"""
import argparse
import json

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from cook_ad.core import OperatingPoint, Trial  # noqa: E402
from cook_ad.data.config import load_config  # noqa: E402
from cook_ad.detectors.hsmm_hindsight import HindsightHSMMDetector  # noqa: E402
from cook_ad.detectors.hsmm_lag import LagHSMMDetector  # noqa: E402
from cook_ad.detectors.hsmm_live import LiveHSMMDetector  # noqa: E402
from cook_ad.detectors.narrate import Lexicon, narrate  # noqa: E402
from cook_ad.detectors.thresholds import BadSets  # noqa: E402
from cook_ad.eval import pools  # noqa: E402
from cook_ad.hsmm.model import HSMM  # noqa: E402
from cook_ad.synthetic import error_injection, generate  # noqa: E402

B = "dataset/processed/breakfast"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=f"{B}/joint_params_train_lexhard.npz")
    ap.add_argument("--trial", type=int, default=0, help="index into the test split")
    ap.add_argument("--inject", choices=error_injection.ERROR_TYPES, default=None)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--kappa", type=float, default=0.5)
    args = ap.parse_args()

    d_max = load_config("configs/breakfast.yaml")["duration"]["d_max_ticks"]
    vocab = json.load(open(f"{B}/vocab.json"))
    hsmm = HSMM.load(args.params, d_max)
    seq = pools.split_sequences(f"{B}/sequences.json", f"{B}/split.json", "test")[args.trial]
    traj = generate.trajectories_from_real_joint(hsmm.params, [seq], d_max)[0]
    if args.inject:
        traj = error_injection.inject(args.inject, traj, np.random.default_rng(0), hsmm.params)
        print(f"injected {args.inject} at ticks {traj['spans']}")
    trial = Trial(seq["trial_id"], np.asarray(traj["verb_ids"]), np.asarray(traj["noun_ids"]))
    lex = Lexicon(vocab, hsmm.params)

    op = OperatingPoint(args.alpha, args.kappa)
    bad = BadSets(hsmm, [args.alpha])
    detectors = {"live": LiveHSMMDetector(hsmm, bad, op), "lag5": LagHSMMDetector(hsmm, bad, op, 5)}
    for det in detectors.values():
        det.reset()
    print(f"\n{trial.trial_id}: {len(trial.verbs)} ticks\n tick  observed            alarms")
    for t, (v, n) in enumerate(zip(trial.verbs, trial.nouns)):
        said = [f"{name}: {a.channel}@{a.tick}" for name, det in detectors.items()
                for a in det.step(int(v), int(n))]
        if said or t == 0 or (v, n) != (trial.verbs[t - 1], trial.nouns[t - 1]):
            print(f"{t:5d}  {lex.phrase(v, n):<18}  {'; '.join(said)}")
    for name, det in detectors.items():
        print(f"  {name} at the end: {[f'{a.channel}@{a.tick}' for a in det.finish()]}")

    hind = HindsightHSMMDetector(hsmm, bad, OperatingPoint(args.alpha))
    detection = hind.detect_trial(trial)
    print(f"\nhindsight: r_hat={detection.trace.r_hat}, {len(detection.trace.segments)} segments, "
          f"{len(detection.alarms)} alarms")
    for q in narrate(detection, hind, vocab):
        print(f"  [{q.tick:4d} {q.channel:<12} {q.severity:<6}] {q.text}")


if __name__ == "__main__":
    main()
