# Huawei Tech Arena 2026, Phase 1

Our solution for the Huawei Ireland Tech Arena 2026 click prediction task. Each event shows a user ten candidate items at
one moment, and the user clicks one of them or none. We predict a click probability for every candidate, and the score is
ROC AUC over all candidate rows together, not per user or per slate.

This repo rebuilds our submissions and their local checks. It's a summary of the work rather than the full experiment
log, so some figures below come from one-off screens whose scripts aren't included.

## Where we got to

| Submission | What it added | Public AUC |
|---|---|---:|
| C1 | User and item vectors, a choice head and a LightGBM residual | 0.823199 |
| C2 | Three of those models with different noise, blended | 0.825478 |
| C3 | Equal weights and three small stack corrections | 0.825805 |
| C4 | A model of what each user gets shown | 0.829778 |
| C4_refined | Graph and rank 16 members, user propensity in the head, tree trained on the whole slate | 0.830118 |
| C5S_c | C4_refined blended with one slate posterior chain | 0.834219 |
| C8 | Half slate posterior chains, half trees that take the chains as input | **0.834673** |

C8 is our best result and C5S_c is the simpler fallback. Both were frozen before they were scored. For reference, an
earlier attempt at this task topped out at 0.817357.

The two final files at all four local cuts:

| Cut | C4_refined | C5S_c | C8 |
|---:|---:|---:|---:|
| 23 Jan | .808906 | .813904 (+.0050) | .814239 (+.0053) |
| 24 Jan | .816428 | .820437 (+.0040) | .820764 (+.0043) |
| 25 Jan | .828568 | .832432 (+.0039) | .832783 (+.0042) |
| 26 Jan | .836849 | .840902 (+.0041) | .841301 (+.0045) |

The local checks predicted the public board well. Over C4_refined, C5S_c gained +.0042 on average locally and +.0041
in public, and C8 gained +.0046 both locally and in public. Earlier, C1 to C3 each scored in public almost exactly .0134
above their cut 24 local AUC.

## The data

The data is synthetic. Training covers 20 to 26 January 2025: 686,824 events, 6.9 million candidate rows, 53,003 users
and 50,000 items. The hidden set is each user's first event on or after 27 January: 52,939 events and 529,390 rows,
running to 2 February.

Every row carries the user's last 20 clicks. Those histories only hold clicks, never the items the user saw and skipped,
and they reach back to 6 January, before the labelled period. A few things about the hidden set shape everything else:

- **Its histories are stale.** Every history in the hidden set stops at the 27 January cutoff, even for events days later.
  The median event comes 17 hours after the cutoff, and about 9% come three or more days after it. We validate on the
  same construction, delays included.
- **Some users are new.** 1,453 hidden set users have no training events, and 294 of those have no history either.
- **No clicks count too.** About 63% of events have a click. Events with none still supply ten negative rows, and about
  40% of the pairs that AUC compares come from them.

Some facts that ruled ideas in or out:

- Items already in the user's history get clicked far more: .29 against .057 for the rest.
- Simple priors are weak. Item popularity alone scores .527, user propensity .539, and both together .547.
- Context does nothing. Time of day, weekday against weekend, row position and day all leave click rates flat.
- A user's click rate varies a lot (sd .157 against .073 from chance alone). Item click rates barely vary beyond chance.
- History order carries no signal. The oldest five clicks predict about as well as the newest five.

## What we learned about the simulator

A lot of the progress came from working out how the data was generated, measured from the files themselves. None of this
was stated by the organisers.

| Part | What we found |
|---|---|
| Preference | User and item vectors sit uniformly on an 8 dimensional sphere. Click log odds are about 6.3 times their dot product plus a constant. A model that knew the true vectors would score about 0.90 |
| Items | Ten categories, which show up as nine strong directions in the offers, with smaller leaves inside them. We resolve 17 leaves at the local cuts and 33 at the final cut |
| Slates | Ten draws from the user's personal mix of categories. About 74% of a slate comes from a stable mix, and 26% echoes the last 20 clicks, each older click weighing about .91 of the next |
| Exposure | What a user is shown says nothing about what they like. Their mix is independent of their preference vector |
| Clicks | A choice between the ten items and an outside option of not clicking, linear in the dot product. On top of that the click rate rises with how much of a category the user is shown: flat near .043 below a share of .06, then roughly linear, .105 at .45 |
| Intercepts | Each user has a click propensity (sd about .7 on the log odds scale) and each item a small one (sd about .2), both Gaussian, plus category and leaf intercepts |
| Feedback | A click raises that category's share in the next slate by .023, fading over about 30 events |
| Absent | No drift, no time of day effect, no fine clusters, no event level intent, no recency in the mix |

The feedback has a sharp consequence. Any feature that counts offers made after a row reads that row's label through the
echo, and an oracle check put that leak at +.009 AUC. We never use offers from after a row.

## How the model evolved

**C1: vectors, a choice head and a tree.** For every day we fit 64 dimensional user and item vectors on all clicks known
before that day, using noise contrastive estimation, with each click counted once even though it repeats across history
snapshots. Each day's table is rotated into common coordinates. At scoring time the user's vector is refreshed from the
event's own history. A shared bilinear head turns vectors into slate probabilities with an outside option, and LightGBM
learns the residual from a few counters. The regularisation alone gave +.045 over our first baseline, refreshing the user
gave +.011, and the tree +.0036.

**C2 and C3: three noise families.** The tables differ only in their noise: item popularity, popularity with the offer
graph in the starting point, and noise from the items each user was actually offered. Blending them gave +.0023 in public.
C3 kept the same members at equal weights and added three small corrections, worth +.0003.

**C4: modelling exposure.** We recovered the ten item categories and their leaves from offers alone, which carry no labels,
then gave head and tree each user's mix, the click lift over their offer share, mix concentration and leaf shares. One
member draws its noise from the exposure model itself, so its vectors hold preference only, and two members are cut down
to their top 8 directions, the preference block. This was the first change to clear our 0.003 bar after we raised it to
that level, at +.0031 to +.0033 locally and +.0040 in public.

**C4_refined: a better stack.** The tree now learns the whole slate's likelihood from the head's utilities rather than
one row at a time, which was the biggest single refinement. The head also sees user propensity. A graph-smoothed table and
a direct rank 16 table joined the members, and the trees went from 300 to 600. Gains fell with the cut, from +.0007
locally to +.0003 in public.

**The turning point.** By then every way of estimating vectors from clicks landed on the same plateau. An exact posterior
sampler and belief propagation matched our tables, so for clicks alone we were at the limit. But none of them ever looks
at the items a user rejected. Earlier attempts to learn from rejections had all memorised the labels, because they fitted
single point estimates. Sampling the full posterior avoids that.

**Slate posterior chains.** `hta.slate` samples every user and item vector at once under the full slate likelihood, on the
Mac's GPU, where one sweep takes .1 s against up to 5 s on the CPU. Scores average the click probability over 2,500
samples, and every scale is fitted inside the sampler. The first research version looked weak, but three defects were
hiding most of its value:

1. 120 samples left too much Monte Carlo noise.
2. Users with no event before the cut were scored from the prior alone. Their query histories now count as evidence, which
   is allowed because they hold only clicks before the cut.
3. Its offsets read offers from after the event, so the echo partly explained the click. They now use offers before the
   row only.

With those fixed the chain alone matched the whole stack. The main upgrades that followed, at cut 24, each built on most
of the ones before it:

| Upgrade | Chain alone |
|---|---:|
| Defects fixed | .816161 |
| Item intercepts | .816824 |
| User covariates for the propensity prior | .817320 |
| Warm-up clicks through the same choice law, with a fitted rival mass | .818359 |
| Click rate as a polynomial in category and leaf share | .819992 |
| Share of each leaf in the user's recent clicks (chain s6) | .820247 |

The warm-up link was the largest step, since warm-up clicks are 70% of the positives at cut 24. The recent leaf share
explained much of what looked like a curved share effect: a short-term interest in recently clicked leaves. Chain s7 is
s6 with each warm-up click's interest term worked out from that user's earlier clicks.

**C5S: chains blended into C4_refined.** C5S_a was the first file to beat C4_refined by more than 0.003. C5S_c uses chain
s6 under two seeds at a share of .75 and keeps the C4_refined trees exactly as they were, which made it the safe fallback.

**C8: chains inside the trees.** C7 feeds the chain's outputs into the head and tree of each of the five tables. A day's
training rows are scored by the chain fitted before that day, with each sample reweighted by the user's earlier outcomes
that day. C8 averages those trees with six chains (s6 under two seeds, s7 under four) at equal weight. That weight was
fixed in advance, and it was also the best of .3 to .7 at every cut.

**Close to the ceiling.** More chains stopped helping, the chain-fed trees add little to the chains, and the old stack
adds nothing on top. Blends of estimators of the same quantity gain about half their shared error, so the small blend
gains left suggest we're near the data's limit.

## How we checked things

- **Honest splits.** A local check at cut d fits on data strictly before 00:00 UTC on day d. It then scores each user's
  first event from d to 27 January, the same construction as the hidden set, with real delays. That's 35,000 to 50,000
  events per cut, and every check keeps all ten candidates, the no click events and minute ties. Early screens used
  22 January, and the final comparisons used all four cuts from 23 to 26 January.
- **Causal tables.** A table fitted at a cutoff already holds the labels of the events before it, so it scores .92 on them
  against .77 the next day. Every training day therefore uses tables fitted strictly before that day, and `lib.leak_gap`
  stops any supervised stage that scores far better on the rows it was fitted on.
- **A strict bar.** A change only moved forward if it beat a matched baseline by more than 0.003 AUC, with a paired
  bootstrap over whole events. A few small steps (C3, C4_refined) went in below it as a deliberate decision.
- **Frozen first.** Weights and settings were fixed before the later cuts and extra seeds were scored, and never tuned to
  the public board.
- **Leak tests.** Changing every later event's slate and outcome moves the exposure inputs by exactly 0. The chain was
  refitted at cut 26 with every later outcome randomised and later histories scrambled, and it still scored .8408 against
  the clean chain's .8406, so it reads nothing from after the cut. Both tests are in `scripts/`.
- **File checks.** Every submission is checked for its header, the exact set of row ids, its row count and value range,
  then read back from disk and compared with the predictions that produced it.

A caveat: every local date was explored at some point, so none is an untouched holdout, and seeds are replications rather
than independent dates. That's why we leaned on agreement across all four cuts and on how well local gains tracked the
public board.

## What didn't work

| Idea | Result |
|---|---|
| Exposure as a direct signal: offer tables, exposure clusters, slate means in the head | Overfit, or +.0004 at most. It only paid once it was modelled as exposure |
| Free user and item parameters trained on the choice likelihood | Memorised: training loss fell from 1.0 to .03 |
| Point estimate tables trained on slate outcomes, from scratch or warm-started | Memorised or flat. Rejections only paid inside a sampled posterior |
| A neural shared-table choice model | +.0004 inside a member, below the bar |
| Exact Newton refits, unit-length vectors, item-side inference, centroid shrinkage, dropout, learning rate and batch sweeps | All on the same plateau or worse |
| Rank 32 and 128 tables | Worse than rank 64 |
| Graph-smoothed table | The largest raw gain of the project (+.009), almost none of which reached the stack. It still earned a small place in C4_refined |
| FISM item-to-item model | Correlated .97 with the main table |
| Extra user, pair and item counters, time of day, weekend, co-clicks | +.0005 at most. Even oracle user and item rates add little |
| Target attention over the history | Identical to its control |
| Belief propagation and Gibbs sampling on clicks alone | Equal to the tables, which is how we found the click-only limit |
| EASE | .55 alone, nothing in a blend |
| CatBoost, a network stacker, one tree over all members | Correlated .995 with LightGBM, or weaker than separate trees |
| A correction fitted after the stack | Double counts clicks the tables already hold |
| More seeds (307 and 401) for the stack | About +.0001 |
| Chain variants: 9 to 12 dimensions, temperature, event slope, exact warm-up integrals | Flat or worse |

A few early negatives turned out to be bugs, such as a head that stopped optimising early. Those were rerun after fixing
and aren't counted against the ideas.

## Running it

You need Python 3.13 and [uv](https://docs.astral.sh/uv/). Put the competition files at `data/train.csv` and
`data/evaluation.csv`. They aren't in this repo because of their size: `train.csv` is 7.26 GB and
`evaluation.csv` 459 MB, far over GitHub's 100 MB file limit. Then:

```sh
uv sync
uv run python -m hta.parse                          # once, about 30 s, writes cache/
uv run python -m hta.portfolio C4_refined 27        # writes and checks submissions/C4_refined.csv
uv run python -m hta.c8                             # rebuilds C8 end to end
uv run python -m hta.c8.c5s_c                       # rebuilds the C5S_c fallback
```

To repeat the local checks at cuts 23 to 26:

```sh
uv run python -m hta.portfolio C4_refined 23,24,25,26               # the baseline, prints each cut's AUC
uv run python -m hta.c8.chains s6 21,22,23,24,25,26 113,211         # chain s6 at every day the trees train on
uv run python -m hta.c8.chains s7 23,24,25,26 113,211,307,401
uv run python -m hta.portfolio C7 23,24,25,26                       # the chain-fed trees
uv run python -m hta.c8.blend 23,24,25,26                           # C8 at each cut
uv run python -m hta.c8.c5s_c 23,24,25,26                           # C5S_c and C4_refined at each cut
```

`uv run python -m hta.compare NEW BASE 23,24` gives the matched gain between two saved runs, with bootstrap bounds.

A few things to know:

- **Hardware.** Everything was built and tested on an Apple Silicon Mac (M1 Max, 64 GB), where the GPU fits run on Metal.
  On other machines, change the `"mps"` device in `pipe.py`, `graph.py` and `slate.py` to `"cuda"` or `"cpu"`. That
  hasn't been tested, and the chains are much slower on the CPU.
- **LightGBM** may fail to import with `Library not loaded: @rpath/libomp.dylib`. Linking `lightgbm/lib/libomp.dylib`
  to the copy inside `torch/lib` in the virtual environment fixes it.
- **Rebuilding C8** takes roughly five to six hours on an M1 Max, and every step can be stopped and resumed. See
  `src/hta/c8/__main__.py` for the faster split version. **C5S_c** takes about two hours, and reuses the two s6 chains at
  cut 27 if a C8 rebuild already made them.
- **Runs don't repeat bit for bit.** The chains and the multi-threaded CPU fits change slightly between runs, even with
  the same seed. A rebuilt file comes out about as different from ours as another seed would: close, but not identical.
- **Threads** are set with `HTA_THREADS`. On the M1 Max, four CPU jobs at two threads each and at most two GPU fits at once
  worked best.

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
| `synth`, `gibbs`, `bp`, `bayes` | A synthetic twin with known truth, and the samplers that found the click-only limit |
| everything else | Ideas we tried and set aside, kept so the results can be rerun |

`scripts/` holds the two leak tests: `exposure_leak_test.py` for the exposure inputs and `chain_leak_test.py` for the
slate chain. Run them with `uv run python scripts/NAME.py`.
