# Huawei Tech Arena 2026, Phase 1

Our solution for the Huawei Ireland Tech Arena 2026 click prediction task. Each event shows a user ten candidate items at
one moment, and the user clicks one of them or none. We predict a click probability for every candidate, and the score is
ROC AUC over all candidate rows together, not per user or per slate.

## Where we got to

| Submission | What it added | Public AUC |
|---|---|---:|
| C1 | User and item vectors, a choice head and a LightGBM residual | 0.823199 |
| C2 | Three of those models blended | 0.825478 |
| C3 | Equal weights and small stack corrections | 0.825805 |
| C4 | A model of what each user gets shown | 0.829778 |
| C4_refined | Graph and rank 16 members, user propensity in the head, tree trained on the whole slate | 0.830118 |
| C5S_c | C4_refined blended with one slate posterior chain | 0.834219 |
| C8 | Half slate posterior chains, half trees that take the chains as input | **0.834673** |

C8 is our best result, and C5S_c is the simpler fallback. Both were frozen before they were scored. Against C4_refined,
on the same local checks, C8 gains between +0.0042 and +0.0053 at each of four cuts and C5S_c between +0.0039 and
+0.0050 (more on the checks below). For reference, an earlier attempt at this task topped out at 0.817357.

## The data

The data is synthetic. Training covers 20 to 26 January 2025: 686,824 events and 6.9 million candidate rows. The hidden
set is each user's first event on or after 27 January, 52,939 events and 529,390 rows. Every row carries the user's last
20 clicks, but those histories only hold clicks, never the items the user saw and skipped. One detail matters a lot. Every
history in the hidden set stops at the 27 January cutoff, even for events several days later, so the model always works
from slightly stale information.

## What we learned about the simulator

A lot of the progress came from working out how the data was generated rather than from bigger models.

- **Preference** is a dot product between user and item vectors on an 8 dimensional sphere, scaled by about 6.3.
  A model that knew the true vectors would reach an AUC of about 0.90.
- **Items** fall into ten categories, with smaller leaves inside each one. We recover them from offers alone, which carry
  no labels.
- **Slates** are drawn from each user's personal mix of categories, plus an echo of their recent clicks. What a user is
  shown says nothing about what they like, so exposure has to be modelled separately from preference.
- **Clicks** follow a choice between the ten items and an "outside option" of not clicking. The click rate also rises with
  how much of a category the user is shown, but not in a simple straight line.
- **Feedback.** A click raises that category's share in the next few slates. Any feature that counts offers made after a
  row leaks that row's label, so we never use one.

## The approach

The final model has two halves.

**Trees fed by learned vectors.** For every day we fit user and item vectors on the clicks known before that day, using
noise contrastive estimation. The variants differ in where the noise comes from: popularity, the user's own offers, or the
exposure model. At scoring time a user's vector is refreshed from
their own history. A shared bilinear choice head turns the vectors into slate probabilities, and LightGBM learns what is
left over, trained on the slate's likelihood rather than on rows one at a time. Five variants of these tables, with two
seeds each, are averaged.

**Slate posterior chains.** The strongest single model samples every user and item vector at once, under the full slate
likelihood, on the Mac's GPU. It also uses the categories, leaves, user and item intercepts, and the share of each leaf in
the user's recent clicks. Every scale is fitted inside the sampler. Scores average the click probability over 2,500
samples. On its own it already beats C4_refined at every local cut.

C8 runs the chains through the trees as extra inputs, then averages those trees with six chains (chain s6 under two seeds,
chain s7 under four) at equal weight. That weight was fixed in advance.

C5S_c is the simpler of the two. It leaves the C4_refined trees exactly as they were and blends them with chain s6 under
two seeds, giving the chain three quarters of the weight. It has no new trees in it, which made it a safe fallback in
case the chain-fed trees didn't carry over to the hidden set.

## How we checked things

- **Honest splits.** Every local check fits on data strictly before a cutoff day, then scores each user's first event
  after it, which is the same construction as the hidden set. We used cutoffs from 23 to 26 January and kept all ten
  candidates, the no click events and the minute ties.
- **A strict bar.** A change only moved forward if it beat a matched baseline by more than 0.003 AUC, with a paired
  bootstrap over whole events.
- **Leak checks.** Supervised stages are checked so that the rows they were fitted on don't score far above rows they
  haven't seen. As a test, the chain was also refitted with every later outcome randomised, and its score didn't change.
- **File checks.** Every submission is checked for its header, the exact set of row ids, its row count and value range,
  then read back from disk and compared with the predictions that produced it.

Plenty of ideas didn't make it. Bigger neural heads, EASE, extra per user parameters, belief propagation and a
correction fitted after the stack all came out flat or worse.

## Running it

You need Python 3.13 and [uv](https://docs.astral.sh/uv/). Put the competition files at `data/train.csv` and
`data/evaluation.csv`, which aren't in this repo (due to very large size), then:

```sh
uv sync
uv run python -m hta.parse                          # once, about 30 s, writes cache/
uv run python -m hta.portfolio C4_refined 27        # writes and checks submissions/C4_refined.csv
uv run python -m hta.c8                             # rebuilds C8 end to end
uv run python -m hta.c8.c5s_c                       # rebuilds the C5S_c fallback
```

A few things to know:

- **LightGBM** may fail to import with `Library not loaded: @rpath/libomp.dylib`. Linking `lightgbm/lib/libomp.dylib`
  to the copy inside `torch/lib` in the virtual environment fixes it.
- **Rebuilding C8** takes roughly five to six hours on an M1 Max, and every step can be stopped and resumed. The chains
  don't repeat bit for bit, even with the same seed, so a rebuilt file comes out close to ours but not identical. See
  `src/hta/c8/__main__.py` for the faster split version.
- **Rebuilding C5S_c** takes about two hours. It needs only the two s6 chains at cut 27, and reuses them if a C8 rebuild
  already made them.
- **Threads** are set with `HTA_THREADS`. On the M1 Max, four CPU jobs at two threads each and at most two GPU fits at once
  worked best.
- **Local checks** at earlier cuts take a cut list, such as `uv run python -m hta.portfolio C4 23,24`, and
  `uv run python -m hta.compare NEW BASE 23,24` gives the matched gain between two runs.

## What's where

| In `src/hta` | What it does |
|---|---|
| `parse`, `lib` | Reads the CSVs into compact arrays, splits, AUC and other shared helpers |
| `states`, `pipe` | User and item vectors, query time inference, cached daily fits |
| `features`, `head`, `model` | Causal counters, the choice head, and one full model from vectors to trees |
| `groups` | Item categories and leaves from offers, user mixes, the exposure model |
| `slate` | The slate posterior sampler on Metal |
| `portfolio`, `final` | Named blends of members, CSV export and the file checks |
| `compare`, `transfer`, `blend` | Local cuts and matched comparisons between runs |
| `c8` | Settings and commands to rebuild C8 and the C5S_c fallback |
| everything else | Ideas we tried and set aside, kept so the results can be rerun |
