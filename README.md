[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Flyrinnn13001/makefun-gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Flyrinnn13001/makefun-gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# Launch-Day Twitch Curve — a pooled-comparables quantile forecaster

An entry for make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by
hour" bounty. `main.py` (Python 3.13, standard library only) reads each case
in the input directory and writes, for every hour 2–24 of the launch day,
five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of that hour's concurrent Twitch
viewers in the game's category.

## The model

Every case carries ~40 past big launches with their full release-day hourly
curves. The program turns each comparable into a log-ratio curve,
`log(viewers_h / viewers_1)` for h = 2..24, anchored on its own first hour —
an empirical answer to "what does a launch day do after hour 1?" The
target's quantiles are its own hour-1 print times the exponential of
per-hour quantiles of that pooled distribution, where each comparable is
weighted by resemblance to the target:

- **Unlock pattern** — rolling midnight-local launches (like GTA VI and the
  practice targets) count double; simultaneous global unlocks count less,
  because their UTC-day shape is different in kind.
- **First-hour scale** — a Gaussian weight on the log distance between the
  comparable's hour-1 viewers and the target's print (bandwidth 1.0), so a
  launch that opened to 800k shapes the forecast of a launch that opened to
  800k more than one that opened to 800.
- **Release-eve ramp** — a Gaussian weight on the log ratio between the last
  hour before the day and hour 1, which separates launches still ramping
  into their day from ones that had already plateaued.

Quantiles are read off the weighted empirical distribution by interpolation
on its midpoint CDF, then steadied with a light [1,2,1] smoothing across
hours. The weights were chosen by leave-one-out validation over the
comparables themselves (each launch forecast from the others): on
rolling-unlock launches with a developed first hour — the regime the
practice and final targets live in — the pooled model scores a mean log
pinball loss of ≈ 0.21–0.26 against ≈ 0.26–0.32 for flat persistence, and
≈ 0.41 against ≈ 0.93 over all launches.

Safety first: any unusable case (bad shape, too few usable comparables, an
internal error) falls back to a persistence forecast around hour 1's print
with a widening spread, so every input always yields a valid output file.
The program is fully deterministic — fixed iteration order, no clocks, no
randomness — and runs a case in about a tenth of a second.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

With no environment set it reads `/input` and writes `/output`, as the
grader runs it (`python3 -P -s /bot/main.py`). Each `<id>.in` produces a
`<id>.out` of the form `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, …]}`.

Data: SullyGnome hourly Twitch category averages, as packaged in the
competition's cases (see the kit's `input/NOTICE.txt`).
