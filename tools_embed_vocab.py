"""Precompute one embedding vector per vocab.json verb/noun and write them to a .npz.

Run under the `embeddings` extra ONLY. `cook_ad` itself never imports an encoder -- it reads
the .npz this writes, which is what keeps the core install lean (pyproject.toml:17).

Breakfast's labels are not plain English -- `egg2plate`, `bunTogether`, `saltnpepper` are
concatenations a sentence encoder tokenizes into garbage subwords. SURFACE_FORM is the
hand-written expansion table; anything absent passes through unchanged.

`stall`/`kitchen` are the SIL tokens. "kitchen" in particular is a real noun that would sit
near every kitchen object and smear the idle state across the vocabulary, so they get an
orthogonal basis direction instead of an embedding (--pin-sil).

    embvenv/bin/python tools_embed_vocab.py --out dataset/processed/breakfast/embeddings.npz
"""
import argparse
import json

import numpy as np

# Concatenated / non-English label tokens -> the phrase actually handed to the encoder.
SURFACE_FORM = {
    # nouns
    "bunTogether": "bun put together",
    "dough2pan": "dough into the pan",
    "egg2pan": "egg into the pan",
    "egg2plate": "egg onto the plate",
    "fruit2bowl": "fruit into the bowl",
    "pancake2plate": "pancake onto the plate",
    "toppingOnTop": "topping on top",
    "saltnpepper": "salt and pepper",
    "cereals": "cereal",
    "eggs": "eggs",
    "powder": "coffee powder",
    "squeezer": "juice squeezer",
    "teabag": "tea bag",
    # verbs
    "stirfry": "stir fry",
    "butter": "butter (spread butter on)",
    "smear": "smear",
}

SIL_TOKENS = {"stall", "kitchen"}


def surface(token):
    return SURFACE_FORM.get(token, token)


def encode(words, model_id, device):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device=device)
    vecs = model.encode([surface(w) for w in words], convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float64)


def pin_sil(words, vecs):
    """Zero each SIL token's encoder vector and give it one extra dimension of its own, so
    cos(SIL, anything real) is exactly 0 -- far colder than any real pair, so no mass leaks
    between the idle state and the real vocabulary."""
    idx = [i for i, w in enumerate(words) if w in SIL_TOKENS]
    if not idx:
        return vecs
    out = np.concatenate([vecs, np.zeros((len(words), len(idx)))], axis=1)
    for slot, i in enumerate(idx):
        out[i, :] = 0.0
        out[i, vecs.shape[1] + slot] = 1.0
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", default="dataset/processed/breakfast/vocab.json")
    ap.add_argument("--out", default="dataset/processed/breakfast/embeddings.npz")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pin-sil", action="store_true", default=True)
    ap.add_argument("--no-pin-sil", dest="pin_sil", action="store_false")
    ap.add_argument("--neighbours", type=int, default=5, help="top-k printed for the sanity gate")
    args = ap.parse_args()

    vocab = json.load(open(args.vocab))
    verbs = [w for w, _ in sorted(vocab["verbs"].items(), key=lambda kv: kv[1])]
    nouns = [w for w, _ in sorted(vocab["nouns"].items(), key=lambda kv: kv[1])]

    verb_vecs, noun_vecs = encode(verbs, args.model, args.device), encode(nouns, args.model, args.device)
    if args.pin_sil:
        verb_vecs, noun_vecs = pin_sil(verbs, verb_vecs), pin_sil(nouns, noun_vecs)

    np.savez(
        args.out, verbs=verb_vecs, nouns=noun_vecs,
        verb_words=np.array(verbs), noun_words=np.array(nouns),
        model_id=np.array(args.model), pin_sil=np.array(args.pin_sil),
    )
    print(f"wrote {args.out}: verbs {verb_vecs.shape} nouns {noun_vecs.shape} model={args.model}\n")

    for name, words, vecs in (("VERBS", verbs, verb_vecs), ("NOUNS", nouns, noun_vecs)):
        sim = vecs @ vecs.T
        np.fill_diagonal(sim, -np.inf)
        print(f"--- {name}: top-{args.neighbours} nearest neighbours ---")
        for i, w in enumerate(words):
            top = np.argsort(-sim[i])[: args.neighbours]
            print(f"  {w:16s} " + "  ".join(f"{words[j]}({sim[i, j]:.2f})" for j in top))
        print()


if __name__ == "__main__":
    main()
