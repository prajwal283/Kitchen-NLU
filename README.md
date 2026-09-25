# DineBot-NLP

**A food-ordering conversational agent where the NLP is the project, not an API call.**

The original [DineBot](https://github.com/krishnaura45/DineBot) wired a Dialogflow
agent to a FastAPI backend and a MySQL database. Dialogflow did the language
understanding; the repository contained a text file of training phrases and a
webhook handler. This version removes Dialogflow entirely and rebuilds the
understanding layer from scratch, so every stage is inspectable, measurable and
defensible.

```
             ORIGINAL                                  THIS PROJECT
  ┌────────────────────────────┐          ┌──────────────────────────────────────┐
  │ Dialogflow (cloud, opaque) │          │ preprocessing  → normalise, expand,  │
  │  · intents                 │          │                  tokenise, POS,      │
  │  · entity lists            │          │                  lemmatise, delexicalise│
  │  · contexts                │  ───►    │ intent         → TF-IDF word+char    │
  └────────────┬───────────────┘          │                  n-grams + linear    │
               │ webhook                  │                  model + rejection   │
               ▼                          │ slots          → linear-chain CRF,   │
  ┌────────────────────────────┐          │                  BIO tagging         │
  │ FastAPI: 4 intent handlers │          │ resolution     → numeral parsing +   │
  │ MySQL (manual setup)       │          │                  fuzzy menu matching │
  └────────────────────────────┘          │ arbitration    → slot-aware rule     │
                                          │                  cascade             │
                                          │ dialogue       → explicit state      │
                                          │                  machine + slot      │
                                          │                  filling + coreference│
                                          │ NLG            → template banks,     │
                                          │                  sentiment-conditioned│
                                          │ SQLite (auto-created)                │
                                          └──────────────────────────────────────┘
```

Nothing calls out to the network. `uvicorn` is the only process you start.

---

## Table of contents

- [What changed, concretely](#what-changed-concretely)
- [Quick start](#quick-start)
- [The NLP pipeline, stage by stage](#the-nlp-pipeline-stage-by-stage)
- [The corpus](#the-corpus)
- [Results](#results)
- [Dialogue management](#dialogue-management)
- [Project layout](#project-layout)
- [API](#api)
- [Optional transformer models](#optional-transformer-models)
- [Limitations and honest caveats](#limitations-and-honest-caveats)
- [Future work](#future-work)
- [References](#references)

---

## What changed, concretely

| | Original | DineBot-NLP |
|---|---|---|
| Intent recognition | Dialogflow console | TF-IDF (word + char n-gram) + linear classifier, trained locally, 4 models compared against a majority baseline |
| Intents | 8 (incl. 2 defaults) | **20** |
| Entity extraction | Dialogflow entity lists | **Linear-chain CRF** over BIO tags, 6 slot types, 24 feature templates |
| Labelled data | ~50 phrases, no entity labels | **3,256 utterances** with intent + BIO spans, generated from a 300-template grammar, plus **2 hand-written evaluation sets** |
| Misspellings | unhandled | fuzzy noisy-channel menu matcher (RapidFuzz + trigram Jaccard) with an accept/suggest threshold |
| Quantities | Dialogflow `@sys.number` | numeral parser: digits, words, `2x`, "a couple of", "half a dozen", Hinglish (`do`, `teen`, `paanch`) |
| Multi-turn memory | Dialogflow contexts | explicit `DialogueState` finite-state machine with pending-question tracking |
| Unclear input | "Sorry I didn't understand" | slot-filling clarification, "did you mean …?", confidence-based rejection with a tuned threshold |
| Pronouns / ellipsis | unsupported | "remove it", "make it 3", "add one" resolved against a focus item |
| Responses | one f-string per intent | template banks with pluralisation, list realisation and sentiment conditioning |
| Evaluation | none | accuracy, macro-F1, per-class report, confusion matrix, entity-level P/R/F1, frame accuracy, 5-fold CV, ablations, noise robustness, latency, error analysis — all in `reports/metrics.json` |
| Database | MySQL, manual dump import, root password in source | SQLite, created and seeded on first run; MySQL still supported via env vars |
| Deployment | ngrok tunnel required | runs entirely locally |
| Observability | none | `/api/nlu` returns the full trace; the demo UI renders it live |

---

## Quick start

```bash
git clone <this repo> && cd DineBot-NLP
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m backend.train            # builds the corpus, trains, evaluates (~3 min)
python reports/make_figures.py     # renders the report figures (optional)

uvicorn backend.main:app --reload
# open http://127.0.0.1:8000
```

The app works before training too — it falls back to a keyword matcher and a
rule-based tagger — but the numbers below need `backend.train` to have run.

Try these in order:

```
hi
what's on the menu
gimme 2 chhole bhatoore n a mango lasi plz      ← typos + chat-speak
mujhe do samosa chahiye                          ← Hinglish
actually remove the samosa
how much is a paneer tikka
add one                                          ← ellipsis, resolved to paneer tikka
is masala dosa vegan
that's all, place the order
track order
40
who won the match yesterday                      ← out of scope, rejected
```

The right-hand panel shows exactly what the pipeline did on each turn:
preprocessing trace, intent posterior, which rule (if any) overrode the
classifier, BIO tags, fuzzy match scores and the dialogue state.

---

## The NLP pipeline, stage by stage

### 1. Preprocessing — `backend/nlp/preprocess.py`

Every transformation is recorded in a trace so it can be shown and explained.

1. **Unicode + whitespace normalisation** (NFKC, curly→straight quotes).
2. **Contraction and chat-speak expansion** — 50 contractions plus a
   domain-specific chat-speak table (`plz→please`, `gimme→give me`, `ordr→order`).
   This is done before tokenisation so `i'd` becomes two tokens, not one unknown.
3. **Regex tokenisation** that keeps digits, hyphenated words and punctuation, and
   is available in two forms: offset-preserving (for the tagger, so BIO labels stay
   aligned) and expansion-applied (for the classifier).
4. **Coarse POS tagging** — a rule tagger (NUM / NOUN / VERB / ADJ / DET / ADP /
   PRON / PUNCT). Chosen over spaCy deliberately: a model download is a dependency
   the demo can fail on, and CRF features only need consistency, not linguistic
   precision.
5. **Lemmatisation** (rule-based, handling the inflections that actually occur here)
   and **Porter stemming** (NLTK, code-only — no corpus downloads).
6. **Stopword removal** against an order-domain stoplist. Note what is *not* a
   stopword: `no`, `not`, `remove`, `without`, `also`, `more`. The standard English
   stoplist destroys the signal that separates `order_add` from `order_remove`, and
   the ablation table shows removing stopwords costs 3.9 macro-F1 points.
7. **Delexicalisation** — menu items become `<food>`, numbers become `<qty>`,
   measure words become `<measure>`:

   ```
   "chuck in 2 samosa and one masala chai"  →  "chuck in <qty> <food> and <qty> <food>"
   "throw in 3 piza"                        →  "throw in <qty> <food>"
   ```

   Both map to the same frame although `chuck in` and `throw in` never co-occur in
   training. The classifier is fed the lemmas **and** the delexicalised frame. This
   single change lifted hand-written held-out accuracy from **56.4% to 78.9%** and
   is the most important design decision in the project.

### 2. Intent classification — `backend/nlp/intent.py`

Features are a union of two TF-IDF views:

- **word 1–2 grams** — phrase cues (`no more`, `how much`, `place order`)
- **char_wb 2–5 grams** — survive misspellings, since `chhole bhatoore` and
  `chole bhature` share most character n-grams

Four models plus a floor are trained on identical features and the identical split:
Logistic Regression, Linear SVM, Complement Naive Bayes, SGD (modified Huber), and
a majority-class `DummyClassifier`. Logistic Regression is what ships — the SVM is
marginally better on macro-F1 but within cross-validation noise, and the softmax
posterior is needed for rejection.

**Rejection.** Every prediction carries a posterior. Below a threshold τ the
prediction is rewritten to `out_of_scope`, so the bot says "I didn't follow that"
rather than confidently executing the wrong action. τ is swept and chosen on the
**dev set only** (`reports/metrics.json → intent.threshold_tuning`); the hard test
set is never used for tuning.

**Interpretability.** `explain_prediction()` returns per-utterance feature
contributions (tf-idf value × class coefficient), and `top_features_for_intent()`
returns the strongest n-grams per class. Both are surfaced in the demo UI.

### 3. Slot filling — `backend/nlp/slots.py`

A **linear-chain CRF** (`sklearn-crfsuite`, L-BFGS, L1+L2) over BIO tags for six
slot types: `FOOD`, `QTY`, `ORDERID`, `DIET`, `SPICE`, `CATEGORY`.

Feature template per token, window −2…+2:

| Group | Features |
|---|---|
| lexical | lowercase form, lemma, stem, 3-char prefix, 2/3-char suffix, forward and backward bigram |
| orthographic | word shape, `is_digit`, digit length, `has_hyphen` |
| syntactic | coarse POS of the token and of each neighbour |
| gazetteer | in-menu-token, starts a multiword dish, ends one, **bucketed fuzzy similarity to the menu** |
| numeric | `is_number_word` (incl. Hinglish), `is_measure_word` |
| positional | BOS, EOS, relative position |

Why a CRF instead of string matching against the menu: the label depends on
context, not just the token.

```
"order 41"                     → 41 is an ORDERID, not a QTY
"2 plates of chole bhature"    → a measure word sits between QTY and FOOD
"remove pizza, keep pav bhaji" → two FOOD spans, different roles
"mujhe do samosa chahiye"      → unseen Hinglish carrier words
"chhole bhatoore"              → misspelled, in no entity list
```

A rule-based gazetteer tagger is kept as both the cold-start fallback and the
baseline row in the results table.

### 4. Lexical resolution — `backend/nlp/lexical.py`

**Quantity parsing.** Digits, `2x`, number words, compound numbers, `a couple of`,
`half a dozen`, `a dozen`, and Hinglish numerals (`ek`, `do`, `teen`, `char`,
`paanch`, `das`).

**Fuzzy menu matching.** A two-stage noisy-channel resolver over the alias
gazetteer: exact alias lookup, then a weighted blend of RapidFuzz
`token_set_ratio`, `partial_ratio`, `WRatio` and a character-trigram Jaccard
tie-breaker. Three outcomes rather than two:

| score | behaviour |
|---|---|
| ≥ 0.78 | accept silently |
| 0.60 – 0.78 | **ask** — "did you mean Chole Bhature?" |
| < 0.60 | reject and show the menu |

The middle band is the point: a wrong silent guess is worse than a question.

### 5. Intent arbitration — `backend/nlp/arbitration.py`

Error analysis on the dev set showed one systematic failure: intents that share a
syntactic frame and differ by a single cue word. A bag-of-n-grams model cannot see
"there is a QTY and a FOOD and no removal cue", but a rule can, and precisely.

So the classifier is followed by a cascade of ten high-precision rules
(`R1`…`R10`), each gated so it may override only when the classifier is not
strongly confident, or when both intents belong to a known-confusable family. Two
lessons are baked in:

- cue matching uses **word boundaries** — plain substring search made the price
  rule fire on "g**rate**ful" and the tracking rule on "veg**eta**rian";
- Hinglish `kar do` is ambiguous ("pack kar do" = please pack; "X ko 2 kar do" =
  change X to 2), so the change reading requires the dative `ko` or emphatic `hi`.

Measured effect (`reports/metrics.json → arbitration`): on the hand-written hard
set, intent accuracy goes from **78.8% to 87.8%** with **16 overrides, 15 of them
correct**; on the grammar test split it costs 0.3 points (95.9% → 95.6%), which is
two utterances. Every override is
reported by name in the API response, so the cascade is auditable rather than
magic.

### 6. Sentiment — `backend/nlp/sentiment.py`

A VADER-style valence lexicon (85 seed terms) with negation, intensifier,
diminisher and punctuation handling, tuned for this domain: `cold`, `late`,
`stale`, `missing` carry food-delivery valence rather than generic valence. It
drives complaint triage (a strongly negative, high-urgency complaint is escalated
rather than given the standard apology) and warmth in response selection.

---

## The corpus

`backend/nlp/dataset.py` builds everything from one markup format:

```
order_add    add [two|QTY] [chhole bhatoore|FOOD]
```

`parse_markup` → text + character spans → `spans_to_bio` → token/BIO pairs. The
same parser reads the generated grammar and the hand-written files, so annotation
can never disagree between them.

| split | n | how it was made | what it is used for |
|---|---|---|---|
| `data/corpus.csv` | 3,256 | ~300 templates × slot inventories, 55% with injected noise | train (80%) / test (20%), stratified, seed 149 |
| `data/dev_set.tsv` | 75 | hand written | threshold tuning, grammar-coverage decisions, rule design |
| `data/hard_test_set.tsv` | 156 | hand written, **never inspected during development** | the honest final number |

**Noise injection** corrupts non-slot text and, deliberately, the inside of `FOOD`
spans too (keyboard-adjacent substitution, transposition, deletion, duplication),
which is what forces the tagger onto context and gazetteer-similarity features
rather than memorised strings.

The hard set contains paraphrases with unseen carrier verbs, real misspellings,
Hinglish code-mixing (`mera order kahan hai`, `poora order cancel kar do`),
elliptical replies, and adversarial near-misses between similar intents.

---

## Results

All numbers are written by `python -m backend.train` into `reports/metrics.json`;
figures are rendered from that file by `reports/make_figures.py`. Regenerate and
the tables below regenerate with them. Values quoted here are from the committed
run — see `reports/metrics.json` for the authoritative set.

### Intent classification (grammar test split, n ≈ 650)

| model | accuracy | macro-F1 | 5-fold CV macro-F1 |
|---|---|---|---|
| majority baseline | 0.068 | 0.006 | — |
| Complement NB | 0.943 | 0.935 | 0.885 |
| SGD (modified Huber) | 0.963 | 0.954 | 0.933 |
| Logistic Regression **(shipped)** | 0.960 | 0.951 | 0.944 |
| Linear SVM | 0.965 | 0.954 | 0.955 |

### Slot filling (entity level — exact span **and** type)

| tagger | micro-F1 | token accuracy |
|---|---|---|
| rule-based gazetteer | 0.852 | 0.958 |
| **linear-chain CRF** | **0.979** | **0.996** |

### The number that matters

| split | intent accuracy | frame accuracy (intent **and** all slots) |
|---|---|---|
| grammar test split | 0.956 | 0.936 |
| hand-written hard set | **0.878** | 0.737 |

The gap between those two rows is the most useful result in the project. Template
data flatters a model: the test split shares a generative grammar with training, so
97% there mostly measures memorisation of the grammar. The hand-written set shares
no sentences with training, and accuracy drops by roughly 8 points. Reporting only
the first number would have been the easy and dishonest choice.

Also measured: noise robustness (clean 0.986 vs noisy 0.938 accuracy — the char
n-grams and the fuzzy matcher earning their place), the feature and preprocessing
ablations, the CRF regularisation sweep, per-intent F1, the confusion matrix,
confident-mistake lists for error analysis, and latency (**18.8 ms mean, 25.5 ms
p95** for the full pipeline on CPU — a Dialogflow round trip is 200–500 ms).

Figures land in `reports/figures/`: confusion matrix, per-intent F1, model
comparison, the generalisation gap, slot scores, the threshold sweep, ablations and
corpus composition.

---

## Dialogue management

`backend/dialogue/` is the explicit replacement for Dialogflow contexts.

**State** (`state.py`): a phase, the cart, a focus item, the pending question, the
last order id and the full turn history.

```
IDLE ──new_order/order_add──► ORDERING ──order_complete──► IDLE
  │                              │
  │                              ├─ dish with no quantity ──► AWAITING_QTY
  │                              ├─ uncertain fuzzy match ──► AWAITING_ITEM_CONFIRM
  │                              └─ order_clear ───────────► AWAITING_CLEAR_CONFIRM
  └──track_order──► AWAITING_ORDER_ID ──id given──► IDLE
```

**Policy** (`policy.py`) maps (state × NLU result) to an action. What it adds:

- **Slot-filling clarification.** "I'd like some samosas" → "How many samosas?" →
  "three" completes the frame. The original returned an error when the counts of
  food items and numbers disagreed.
- **Uncertain-match confirmation** before committing a dish to the cart.
- **Pronoun and ellipsis resolution.** "remove it", "make it 3", "add one",
  "another one" resolve against the focus item — a minimal coreference layer over
  the cart.
- **Confirmation before destruction.** Clearing the cart asks first, and a "no"
  restores it.
- **Yes/no bound to the last question**, so `affirm`/`deny` actually mean
  something. The same "yes" places an order, confirms a dish or confirms a wipe,
  depending on `pending_question`.
- **Derailment recovery.** If the bot asked for an order id and the user instead
  says "make it 3 lassi", the number is a quantity, not an id — the question is
  dropped and the order intent honoured.
- **Sentiment-conditioned complaints.**

**NLG** (`nlg.py`): template banks per action with random selection, English
pluralisation that knows `Mango Lassi` has no plural, a proper comma/"and" list
joiner, and warmth conditioned on sentiment.

Every turn is logged to the `conversation_log` table with its intent, confidence
and slots — which is how you would harvest real utterances to retrain on later.

---

## Project layout

```
DineBot-NLP/
├── backend/
│   ├── main.py                 FastAPI app: /api/chat, /api/nlu, /api/metrics, …
│   ├── train.py                trains + evaluates everything, writes reports/
│   ├── db.py                   SQLite layer (auto-created, MySQL-switchable)
│   ├── nlp/
│   │   ├── preprocess.py       normalisation, tokenisation, POS, lemma, stem
│   │   ├── menu.py             menu knowledge base + alias gazetteer
│   │   ├── lexical.py          numeral parsing, fuzzy matching, delexicalisation
│   │   ├── dataset.py          corpus grammar, BIO annotation, noise injection
│   │   ├── intent.py           TF-IDF + linear models, rejection, explanations
│   │   ├── slots.py            CRF tagger, features, rule baseline, resolution
│   │   ├── arbitration.py      slot-aware rule cascade
│   │   ├── sentiment.py        lexicon sentiment + urgency
│   │   ├── transformer_intent.py  optional MiniLM / DistilBERT heads
│   │   ├── nlu.py              the pipeline facade → NLUResult
│   │   └── models/             trained artefacts (.joblib)
│   └── dialogue/
│       ├── state.py            dialogue state + session store
│       ├── policy.py           state × intent → action
│       └── nlg.py              response templates + surface realisation
├── data/
│   ├── corpus.csv              generated corpus (intent + BIO)
│   ├── dev_set.tsv             hand-written tuning set
│   ├── hard_test_set.tsv       hand-written held-out set
│   └── dinebot.db              SQLite, created on first run
├── notebooks/
│   ├── 01_nlp_evaluation.ipynb      dataset stats, metrics, errors, ablations
│   └── 02_transformer_benchmark.ipynb  MiniLM / DistilBERT vs the baseline
├── reports/
│   ├── metrics.json            every number, machine-readable
│   ├── predictions_*.csv       per-utterance predictions for error analysis
│   ├── make_figures.py         renders figures/ from metrics.json
│   └── figures/
├── frontend/index.html         demo site + live NLU inspector
├── docs/DEMO_SCRIPT.md         run sheet for the presentation
├── legacy/                     the original MySQL dump and Dialogflow phrases
└── requirements.txt
```

---

## API

| method | path | purpose |
|---|---|---|
| POST | `/api/chat` | one turn: `{message, session_id}` → reply + action + full NLU trace + dialogue state |
| POST | `/api/nlu` | parse only, no side effects — for the inspector and for demos |
| GET | `/api/menu` | menu from the database |
| GET | `/api/order/{id}` | status, line items and total |
| GET | `/api/session/{sid}` | dialogue state and turn history |
| POST | `/api/session/{sid}/reset` | drop a session |
| GET | `/api/model-info` | loaded models, threshold, intent list |
| GET | `/api/metrics` | `reports/metrics.json` |
| GET | `/api/log` | recent conversation log rows |
| POST | `/webhook/dialogflow` | accepts the original Dialogflow webhook body, so an existing agent can still point here |

Interactive docs at `/docs` (FastAPI's OpenAPI UI).

---

## Optional transformer models

`backend/nlp/transformer_intent.py` adds two transformer variants. Both are
**optional** — the app checks whether the libraries and weights are present and
silently falls back to TF-IDF, so a machine with no GPU, no `torch` or no internet
still runs the whole demo.

```bash
pip install torch sentence-transformers transformers

python -m backend.nlp.transformer_intent --train-embed       # MiniLM + logistic head
python -m backend.nlp.transformer_intent --train-distilbert  # fine-tuned DistilBERT
```

| variant | what it is | cost |
|---|---|---|
| **A — frozen MiniLM + linear probe** | `all-MiniLM-L6-v2` sentence embeddings → Logistic Regression | ~90 s on CPU, ~22 MB, runs inside the API process |
| **B — fine-tuned DistilBERT** | `distilbert-base-uncased` + classification head, 3 epochs | ~4 min on a Colab T4, ~260 MB |

Once variant A is trained a "transformer head" toggle appears in the demo header
and `/api/chat` accepts `use_transformer: true`, so you can switch backends live
and compare. `SemanticMenuSearch` uses the same encoder to answer queries like
"something cold and sweet" against the dish descriptions — retrieval that
bag-of-n-grams cannot do at all.

`notebooks/02_transformer_benchmark.ipynb` runs the comparison and fills in the
results table. It is written for Colab, where `torch` and Hugging Face access are
already available.

> The committed `reports/metrics.json` contains the classical numbers only,
> because the environment it was produced in had no access to the Hugging Face
> model hub. Run the notebook to add the transformer rows — the expected win is on
> paraphrase generalisation (the hard set), not on the grammar split, where TF-IDF
> is already near ceiling.

---

## Limitations and honest caveats

Worth stating plainly, because they are the questions an examiner will ask.

1. **The training corpus is synthetic.** It comes from a template grammar, so the
   grammar test split measures grammar memorisation more than language
   understanding. That is exactly why the hand-written hard set exists and why both
   numbers are reported side by side.
2. **The hard set is small** (156 utterances) and written by one person, so per-intent
   estimates on it are noisy. It is a sanity check on generalisation, not a benchmark.
3. **Twenty intents is still a closed world.** Anything outside them is at best
   rejected as `out_of_scope`; nothing is learned online.
4. **The rule cascade is hand-tuned on the dev set.** It is measured on the hard
   set and helps there with zero harmful overrides, but rules are maintenance debt:
   every new intent risks interacting with them.
5. **Hinglish support is lexical, not linguistic.** Numerals, some carrier verbs and
   particles are covered; there is no morphological analysis of Hindi.
6. **Sentiment is a hand-built lexicon**, evaluated only qualitatively. There is no
   labelled sentiment test set in this project.
7. **Sessions live in process memory.** Restarting the server loses open carts;
   production would need Redis.
8. **No speech, no images, no payments, no authentication.**

---

## Future work

- Replace the synthetic grammar with mined real utterances from `conversation_log`
  and re-measure the generalisation gap.
- Joint intent + slot modelling (a shared encoder with two heads) instead of two
  independent models — the tasks inform each other.
- Learn the dialogue policy from transcripts rather than hand-writing it.
- Active learning: surface the low-confidence and rule-overridden turns for
  annotation, retrain nightly.
- Multilingual support with a genuine Hindi/Hinglish tokeniser and transliteration.
- Speech in and out for a kiosk deployment.

---

## References

- Lafferty, McCallum & Pereira (2001), *Conditional Random Fields: Probabilistic
  Models for Segmenting and Labeling Sequence Data*
- Sang & De Meulder (2003), *CoNLL-2003 Shared Task* — the BIO tagging scheme and
  entity-level evaluation protocol used here
- Hutto & Gilbert (2014), *VADER: A Parsimonious Rule-based Model for Sentiment
  Analysis* — the design the sentiment module follows
- Henderson, Thomson & Williams (2014), *The Second Dialog State Tracking Challenge*
  — frame accuracy as a joint metric
- Sanh et al. (2019), *DistilBERT*; Reimers & Gurevych (2019), *Sentence-BERT*
- Larson et al. (2019), *An Evaluation Dataset for Intent Classification and
  Out-of-Scope Prediction* — why rejection needs its own metric
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft), chs. 8
  (sequence labelling), 15 (dialogue systems)
- Original project: [krishnaura45/DineBot](https://github.com/krishnaura45/DineBot)
  (Apache 2.0) — menu, prices, schema and the two seed orders are carried over from it.

---

Licensed Apache 2.0, following the original project.
