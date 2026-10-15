"""PREGO's trimmed, step-level protocol (metric E) as scripts/eval/eval_epic_tent.py --trim runs it.

Each test pins one decision the protocol could quietly get wrong. The reference is PREGO's own
scorer (step_anticipation/src/models/llm_hf.py): `correct = matches[:-1]; mistake = matches[-1]`,
one positive per trimmed video, every earlier step a negative.
"""
import importlib.util
from pathlib import Path

import pytest

from cook_ad.eval import metric_e
from cook_ad.eval.metric_g import Flag
from cook_ad.llm import predict

REPO_ROOT = Path(__file__).resolve().parent.parent
GROUP = ("transposition", "omission")
VERBS = {0: "stall", 1: "open", 2: "spread", 3: "insert"}
NOUNS = {0: "idle", 1: "tentbag", 2: "tent", 3: "stake"}
SIL = ("stall", "idle")


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("eval_epic_tent",
                                                  REPO_ROOT / "scripts" / "eval" / "eval_epic_tent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.step_extents = metric_e.step_extents
    module.drop_truncated_under_run = metric_e.drop_truncated_under_run
    return module


class _Lex:
    def verb(self, i):
        return VERBS[int(i)]

    def noun(self, i):
        return NOUNS[int(i)]


def _traj(runs):
    """runs: [(verb_id, noun_id, n_ticks)] -> a trajectory-shaped dict."""
    verbs, nouns = [], []
    for v, n, d in runs:
        verbs += [v] * d
        nouns += [n] * d
    return {"trial_id": "p99", "verb_ids": verbs, "noun_ids": nouns}


def _truth(n_ticks, *records):
    return {"n_ticks": n_ticks,
            "records": [{"window": list(w), "error_type": t} for w, t in records]}


# open tentbag 0-4 | idle 5-6 | spread tent 7-11 | insert stake 12-19
RUNS = [(1, 1, 5), (0, 0, 2), (2, 2, 5), (3, 3, 8)]


def test_trim_cuts_after_the_last_tick_of_the_earliest_in_group_mistake(ev):
    seq = _traj(RUNS)
    truth = _truth(20, ((12, 15), "transposition"), ((7, 9), "omission"))
    (trimmed, window), = ev.trim_at_first_mistake([seq], [truth], GROUP)
    assert window == (7, 9), "earliest ONSET wins, not first in the record list"
    assert len(trimmed["verb_ids"]) == 10


def test_out_of_group_errors_neither_trigger_the_cut_nor_survive(ev):
    """An earlier misuse is relabelled normal, exactly as PREGO counts it a correct step."""
    truth = _truth(20, ((1, 3), "substitution"), ((12, 15), "transposition"))
    (trimmed, window), = ev.trim_at_first_mistake([_traj(RUNS)], [truth], GROUP)
    assert window == (12, 15)


def test_a_session_with_no_in_group_mistake_is_kept_whole(ev):
    """PREGO: videos 'representing correct procedures are maintained unaltered'."""
    seq = _traj(RUNS)
    (trimmed, window), = ev.trim_at_first_mistake([seq], [_truth(20, ((1, 3), "stall"))], GROUP)
    assert window is None and trimmed is seq


def test_step_extents_drop_idle_and_charge_it_to_the_preceding_action(ev):
    """PREGO's steps are actions only. The idle run 5-6 belongs to `open tentbag`."""
    assert ev.step_extents(_traj(RUNS), _Lex(), SIL) == [(0, 6), (7, 11), (12, 19)]


def test_an_action_resuming_after_idle_is_the_same_step(ev):
    """PREGO's gt never repeats an action back to back, so a pause does not start a new step."""
    runs = [(1, 1, 3), (0, 0, 2), (1, 1, 3), (2, 2, 4)]
    assert ev.step_extents(_traj(runs), _Lex(), SIL) == [(0, 7), (8, 11)]


def test_leading_idle_is_charged_to_the_first_action(ev):
    assert ev.step_extents(_traj([(0, 0, 3), (1, 1, 4)]), _Lex(), SIL) == [(0, 6)]


def test_score_counts_like_preGOs_scorer(ev):
    """Mistake on the last step (12-19): one positive. Two earlier steps: negatives."""
    traj = _traj(RUNS)
    trimmed = [(traj, (12, 19))]
    truth = [_truth(20, ((12, 19), "transposition"))]
    hit_and_false_alarm = [Flag(5, 1.0, "x"), Flag(13, 1.0, "x")]   # 5 is in the idle -> step 0
    r = ev.score_prego([hit_and_false_alarm], trimmed, [traj], _Lex(), SIL, truth, GROUP)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 0, 1, 1)
    assert r["per_type"]["transposition"]["recall"] == 1.0


def test_a_mistake_spanning_several_steps_is_still_one_positive(ev):
    traj = _traj(RUNS)
    truth = [_truth(20, ((9, 14), "transposition"))]
    r = ev.score_prego([[Flag(8, 1.0, "x")]], [(traj, (9, 14))], [traj], _Lex(), SIL, truth,
                       GROUP)
    assert r["tp"] + r["fn"] == 1, "steps 7-11 and 12-19 both overlap it: ONE positive"
    assert r["tp"] == 1 and r["fp"] == 0 and r["tn"] == 1


def test_a_miss_is_a_false_negative_and_quiet_negatives_are_true_negatives(ev):
    traj = _traj(RUNS)
    truth = [_truth(20, ((12, 19), "omission"))]
    r = ev.score_prego([[]], [(traj, (12, 19))], [traj], _Lex(), SIL, truth, GROUP)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (0, 1, 0, 2)
    assert r["per_type"]["omission"]["recall"] == 0.0


def test_only_the_truncated_final_steps_under_run_is_dropped(ev):
    """The cut step never closed, so 'it ended too early' is an artifact of trimming. Over-runs,
    and under-runs of steps that DID close, are kept."""
    traj = _traj(RUNS)
    flags = [Flag(19, 1.0, predict.DUR_SHORT), Flag(11, 1.0, predict.DUR_SHORT),
             Flag(19, 1.0, predict.DUR_LONG), Flag(12, 1.0, predict.SYMBOL)]
    kept = ev.drop_truncated_under_run(flags, traj, _Lex())
    assert Flag(19, 1.0, predict.DUR_SHORT) not in kept
    assert len(kept) == 3


def test_balanced_precision_scores_chance_at_one_half(ev):
    """TI-PREGO's Prec_b = TP / (TP + (N+/N-) FP) = TPR / (TPR + FPR). Flagging every step is
    TPR = FPR = 1, so it scores 0.5 and F1_b 2/3 -- the property that makes their 20-40%
    Epic-tent-O numbers sit BELOW this null rather than above PREGO's."""
    traj = _traj(RUNS)
    truth = [_truth(20, ((12, 19), "transposition"))]
    every_step = [Flag(t, 1.0, "x") for t in (0, 7, 12)]
    r = ev.score_prego([every_step], [(traj, (12, 19))], [traj], _Lex(), SIL, truth, GROUP)
    assert (r["tp"], r["fp"]) == (1, 2)
    assert r["precision"] == pytest.approx(1 / 3)
    assert r["precision_b"] == pytest.approx(0.5)
    assert r["f1_b"] == pytest.approx(2 / 3)


def test_untrimmed_step_fpr_counts_correct_steps_only(ev):
    """The untrimmed step FPR uses the trimmed protocol's unit. Steps: open tentbag 0-6 (idle
    charged to it), spread tent 7-11, insert stake 12-19. A mistake on spread tent removes that
    step from the denominator; an out-of-group span does not (it was relabelled normal)."""
    traj = _traj(RUNS)
    spans = [((8, 9), "transposition")]
    flags = [Flag(tick=6, channel="x"), Flag(tick=9, channel="x")]
    assert ev.step_false_alarms(flags, traj, _Lex(), SIL, spans) == (1, 2)
    assert ev.step_false_alarms([], traj, _Lex(), SIL, spans) == (0, 2)
    # Out-of-group types never reach `spans` (group_truth drops them), so all 3 steps count.
    truth = _truth(20, ((8, 9), "substitution"))
    in_group, _ = ev.group_truth(truth, GROUP)
    assert ev.step_false_alarms(flags, traj, _Lex(), SIL, in_group) == (2, 3)
