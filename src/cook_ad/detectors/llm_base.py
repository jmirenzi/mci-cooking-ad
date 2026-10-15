"""class LLMDetector: what the two LLM detectors share — the system prompt (docs/llm.md §2).

The prompt bytes are part of the response cache key (sha256 of base_url, model, messages and
temperature), so they must not change: every string here comes from llm/prompts.py unchanged.
"""
from cook_ad.core import AnomalyDetector, Trial
from cook_ad.llm import predict, prompts, textify


class LLMDetector(AnomalyDetector):
    """An LLM queried through `client`, with a system prompt built by fit()."""
    protocol = "incremental"

    def __init__(self, client, vocab, lexicon, variant="no-recipes", labels=None, n_shot=0,
                 shot_select="random", shot_cot=False, seed=0, rationale_client=None,
                 prompt_unit="step", **prompt_kwargs):
        self.client, self.vocab, self.lexicon = client, vocab, lexicon
        self.variant, self.labels = variant, labels
        self.n_shot, self.shot_select, self.shot_cot, self.seed = n_shot, shot_select, shot_cot, seed
        self.rationale_client = rationale_client or client
        self.prompt_unit = prompt_unit          # the unit argument the prompt is built with
        self.prompt_kwargs = prompt_kwargs      # domain, sil tokens, canonical order (EPIC-Tent)
        self.system_prompt = None
        self.exemplars, self.rationales = [], None

    def fit(self, train: list[Trial]) -> None:
        """Few-shot exemplars drawn from `train` (and Auto-CoT rationales), then the prompt."""
        if self.n_shot > 0:
            shots = [{"verb_ids": t.verbs, "noun_ids": t.nouns} for t in train]
            self.exemplars = prompts.select_exemplars(shots, self.lexicon, self.n_shot, self.seed,
                                                      method=self.shot_select, vocab=self.vocab)
            if self.shot_cot:
                self.rationales = predict.build_rationales(self.rationale_client, self.exemplars,
                                                           prompts.exemplar_boundary)
        self.build_prompt()

    def build_prompt(self):
        """The system prompt from the current exemplars and rationales."""
        self.system_prompt = prompts.build_variant(
            self.variant, self.vocab, self.labels, self.protocol, self.prompt_unit,
            exemplars=self.exemplars, rationales=self.rationales, **self.prompt_kwargs)
        return self.system_prompt

    def _prompt(self):
        if self.system_prompt is None:
            raise RuntimeError(f"{self.name}: call fit() first (it builds the system prompt)")
        return self.system_prompt

    def steps(self, verbs, nouns):
        """Runs of constant (verb, noun), as the LLM sees them."""
        return textify.steps_from_ids(verbs, nouns, self.lexicon)
