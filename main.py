"""Hour-by-hour Twitch viewer forecast for a big game launch day.

Written for make.fun's gta6-twitch-ccu-forecast-01. The program reads every
case file in the input directory and writes one forecast per case: for each
hour 2..24 of the launch day, five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of
that hour's concurrent Twitch viewers in the game's category.

Model: pooled launch curves. Every comparable in the case is a past launch
with its full release-day hourly curve. For each one, form the log-ratio
curve log(viewers_h / viewers_1) anchored on its own first hour; the pool of
those curves is an empirical distribution of "what a launch day does after
hour 1". The target's forecast is hour 1's print times the exponential of
per-hour quantiles of that pool, with each comparable weighted by how much
it resembles the target: same unlock pattern (rolling midnight-local vs one
global moment), similar first-hour scale, and a similar pre-release ramp in
the hours just before the day starts. The weighted quantiles are blended
with the plain pooled quantiles and lightly smoothed across hours, which
keeps the tails honest without chasing noise.

Safety: any problem with a case (bad shape, empty pool, an internal error)
falls back to a flat persistence forecast around hour 1's print with a
widening log-normal spread, so every case file always gets a valid output.
Everything is deterministic: fixed iteration order, no clocks, no randomness.

Run by the grader as `python3 -P -s /bot/main.py`; locally, set
MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR to stand in for /input and /output.
Standard library only.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
FALLBACK_SIGMA = 0.15

HOURS = list(range(2, 25))

# Weighting knobs, tuned by leave-one-out over the comparables themselves.
ROLLING_MATCH = 2.0      # comparable shares the target's unlock pattern
ROLLING_MISMATCH = 0.7   # it does not
SCALE_BANDWIDTH = 1.0    # log units; closeness of first-hour viewers
RAMP_BANDWIDTH = 0.9     # log units; closeness of the pre-release ramp ratio
EVE_BANDWIDTH = 0.0      # log units; closeness of the whole release-eve curve (0 = off)
VPC_BANDWIDTH = 0.0      # log units; closeness of hour-1 viewers per channel (0 = off)
GROWTH_BANDWIDTH = 0.0   # log units; closeness of release-eve growth into hour 1 (0 = off)
EXCLUDE_EARLY_ACCESS = 0  # 1 drops comparables whose "launch day" followed an early-access window
SMOOTH = 1               # 1 applies the light across-hour smoothing of the quantiles
MIN_WEIGHT = 0.01        # nobody is fully excluded by the soft weights
POOL_BLEND = 0.0         # share of the unweighted pooled quantiles in the mix
EVE_HOURS = list(range(-13, 0))  # the release eve: first unlocks to day start


def epoch(text):
    """Seconds since the epoch for an ISO 8601 timestamp, or None."""
    if not isinstance(text, str):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def round2(x):
    """Round half up to 2 decimals; whole numbers stay whole for compact JSON."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def is_launch(comp):
    """True for a real launch comparable; the target category's own pre-release
    history rides along in every case and is not a launch."""
    name = str(comp.get("name", ""))
    unlock = str(comp.get("unlock", ""))
    if "category before release" in name:
        return False
    if unlock.startswith("Not a launch"):
        return False
    return True


def is_rolling(text):
    """An unlock pattern is 'rolling' when the game unlocks at midnight local
    time territory by territory, as opposed to one global moment."""
    t = str(text).lower()
    if "midnight local" in t or "rolling" in t:
        return True
    if "simultaneous" in t or "global unlock" in t or "one global" in t:
        return False
    return True


def hourly_by_offset(points, start_epoch):
    """Map whole-hour offsets from start_epoch to viewers for a point list
    whose rows carry a `from` (or `at`) timestamp and a `viewers` count."""
    out = {}
    if start_epoch is None or not isinstance(points, list):
        return out
    for row in points:
        if not isinstance(row, dict):
            continue
        t = epoch(row.get("from", row.get("at")))
        v = row.get("viewers")
        if t is None or not isinstance(v, (int, float)) or v is None:
            continue
        offset = int(round((t - start_epoch) / 3600.0))
        out[offset] = float(v)
    return out


class CompCurve:
    """One comparable launch: its release-day log-ratio curve and features."""

    def __init__(self, comp):
        self.ok = False
        start = epoch(comp.get("day_start"))
        if start is None:
            return
        by_hour = hourly_by_offset(comp.get("hourly"), start)
        # Hour 1 of the day is offset 0.
        v1 = by_hour.get(0)
        if not v1 or v1 <= 0:
            return
        self.v1 = float(v1)
        self.log_v1 = math.log(self.v1)
        # Hour 1's viewers per live channel, a fingerprint of how the first
        # hour was watched (a few big streams vs a wide field of small ones).
        self.log_vpc = None
        first_row = None
        for row in comp.get("hourly") or []:
            if isinstance(row, dict) and epoch(row.get("from")) == start:
                first_row = row
                break
        if first_row is not None:
            ch = first_row.get("channels")
            if isinstance(ch, (int, float)) and ch and ch > 0:
                self.log_vpc = math.log(self.v1 / float(ch))
        self.rolling = is_rolling(comp.get("unlock", ""))
        # Pre-release ramp: the last hour before the day, relative to hour 1.
        pre = by_hour.get(-1)
        self.pre_ratio = (pre / self.v1) if pre and pre > 0 else None
        # The release-eve curve: log viewers relative to hour 1, hourly over
        # the 13 hours before the day (the window in which a rolling launch's
        # first territories unlock). Two launches whose eves match in shape
        # tend to share the day that follows.
        self.eve = {}
        for off in EVE_HOURS:
            v = by_hour.get(off)
            if v and v > 0:
                self.eve[off] = math.log(v / self.v1)
        # Release-eve growth: how much the audience grew over the last seven
        # hours before the day, in log units. A launch still climbing into
        # hour 1 has a different day ahead of it than one already plateaued.
        eve_vals = [by_hour.get(off) for off in range(-7, -1)]
        eve_vals = [v for v in eve_vals if v and v > 0]
        self.growth = None
        if len(eve_vals) >= 3:
            self.growth = math.log(self.v1) - math.log(sum(eve_vals) / len(eve_vals))
        self.early_access = "early access" in str(comp.get("unlock", "")).lower()
        # Log-ratio curve for hours 2..24 (offsets 1..23), gaps interpolated.
        raw = {}
        for h in HOURS:
            v = by_hour.get(h - 1)
            if v and v > 0:
                raw[h] = math.log(v / self.v1)
        if len(raw) < 15:
            return
        self.ratio = fill_gaps(raw)
        self.ok = True


def fill_gaps(raw):
    """Fill missing hours of a log-ratio curve by linear interpolation
    between the nearest present neighbours; carry the nearest value outward."""
    hours = sorted(raw)
    out = {}
    for h in HOURS:
        if h in raw:
            out[h] = raw[h]
            continue
        before = max((x for x in hours if x < h), default=None)
        after = min((x for x in hours if x > h), default=None)
        if before is not None and after is not None:
            frac = (h - before) / (after - before)
            out[h] = raw[before] + frac * (raw[after] - raw[before])
        elif before is not None:
            out[h] = raw[before]
        elif after is not None:
            out[h] = raw[after]
    return out


def weighted_quantile(pairs, tau):
    """Quantile of weighted (value, weight) pairs: linear interpolation on
    the midpoint-weighted empirical CDF, clamped to the observed range."""
    pts = sorted(pairs)
    total = sum(w for _, w in pts)
    if total <= 0:
        return pts[len(pts) // 2][0]
    # CDF ordinates at each point's midpoint mass.
    xs = []
    fs = []
    acc = 0.0
    for value, w in pts:
        acc += w
        xs.append(value)
        fs.append((acc - 0.5 * w) / total)
    if tau <= fs[0]:
        return xs[0]
    if tau >= fs[-1]:
        return xs[-1]
    for i in range(1, len(fs)):
        if tau <= fs[i]:
            span = fs[i] - fs[i - 1]
            frac = 0.0 if span <= 0 else (tau - fs[i - 1]) / span
            return xs[i - 1] + frac * (xs[i] - xs[i - 1])
    return xs[-1]


def persistence(p1):
    """The safe forecast: every hour at hour 1's print, spread widening like
    a random walk's. Used whenever anything about a case is unusable."""
    rows = []
    base = max(1.0, p1)
    for h in HOURS:
        spread = FALLBACK_SIGMA * math.sqrt(h - 1)
        q = [round2(base * math.exp(z * spread)) for z in Z]
        rows.append({"hour": h, "q": q})
    return {"hours": rows}


def target_features(case, p1):
    """What the target looks like, in the same terms as a comparable's:
    its unlock pattern, its last pre-day hour relative to hour 1, its
    release-eve curve from the case's own hourly history, and hour 1's
    viewers per channel from the known row."""
    notes = str(case.get("notes", ""))
    rolling = is_rolling(notes)
    start = epoch(case.get("day_start"))
    pre_ratio = None
    eve = {}
    if start is not None and p1 > 0:
        by_hour = hourly_by_offset(case.get("history_hourly"), start)
        pre = by_hour.get(-1)
        if pre and pre > 0:
            pre_ratio = pre / p1
        for off in EVE_HOURS:
            v = by_hour.get(off)
            if v and v > 0:
                eve[off] = math.log(v / p1)
    log_vpc = None
    row = hour_one_row(case)
    if row is not None:
        ch = row.get("channels")
        if isinstance(ch, (int, float)) and ch and ch > 0 and p1 > 0:
            log_vpc = math.log(p1 / float(ch))
    growth = None
    if start is not None and p1 > 0:
        by_hour = hourly_by_offset(case.get("history_hourly"), start)
        eve_vals = [by_hour.get(off) for off in range(-7, -1)]
        eve_vals = [v for v in eve_vals if v and v > 0]
        if len(eve_vals) >= 3:
            growth = math.log(p1) - math.log(sum(eve_vals) / len(eve_vals))
    return rolling, pre_ratio, eve, log_vpc, growth


def eve_distance(curve_eve, target_eve):
    """Root-mean-square gap between two release-eve curves over the hours
    both have; None when they share fewer than four hours."""
    shared = [off for off in EVE_HOURS if off in curve_eve and off in target_eve]
    if len(shared) < 4:
        return None
    sq = sum((curve_eve[o] - target_eve[o]) ** 2 for o in shared)
    return math.sqrt(sq / len(shared))


def comp_weight(curve, log_p1, rolling, pre_ratio, eve, log_vpc, growth):
    w = ROLLING_MATCH if curve.rolling == rolling else ROLLING_MISMATCH
    d_scale = (curve.log_v1 - log_p1) / SCALE_BANDWIDTH
    w *= math.exp(-0.5 * d_scale * d_scale)
    if pre_ratio is not None and curve.pre_ratio is not None:
        d_ramp = (math.log(curve.pre_ratio) - math.log(pre_ratio)) / RAMP_BANDWIDTH
        w *= math.exp(-0.5 * d_ramp * d_ramp)
    if EVE_BANDWIDTH > 0 and eve:
        dist = eve_distance(curve.eve, eve)
        if dist is not None:
            d_eve = dist / EVE_BANDWIDTH
            w *= math.exp(-0.5 * d_eve * d_eve)
    if VPC_BANDWIDTH > 0 and log_vpc is not None and curve.log_vpc is not None:
        d_vpc = (curve.log_vpc - log_vpc) / VPC_BANDWIDTH
        w *= math.exp(-0.5 * d_vpc * d_vpc)
    if GROWTH_BANDWIDTH > 0 and growth is not None and curve.growth is not None:
        d_gro = (curve.growth - growth) / GROWTH_BANDWIDTH
        w *= math.exp(-0.5 * d_gro * d_gro)
    return max(w, MIN_WEIGHT)


def smooth(curve_by_tau):
    """A light [1,2,1]/4 moving average over hours for each quantile level,
    which steadies the tails; the ends use the smaller window they have."""
    out = {}
    for tau, values in curve_by_tau.items():
        sm = []
        for i, v in enumerate(values):
            acc = 2.0 * v
            weight = 2.0
            if i > 0:
                acc += values[i - 1]
                weight += 1.0
            if i < len(values) - 1:
                acc += values[i + 1]
                weight += 1.0
            sm.append(acc / weight)
        out[tau] = sm
    return out


def model_forecast(case, p1):
    """The pooled-curve forecast for one case around its hour-1 print p1."""
    curves = []
    for comp in case.get("comparables") or []:
        if isinstance(comp, dict) and is_launch(comp):
            curve = CompCurve(comp)
            if curve.ok:
                curves.append(curve)
    if EXCLUDE_EARLY_ACCESS:
        fresh = [c for c in curves if not c.early_access]
        if len(fresh) >= 5:
            curves = fresh
    if len(curves) < 5:
        return persistence(p1)
    rolling, pre_ratio, eve, log_vpc, growth = target_features(case, p1)
    log_p1 = math.log(max(1.0, p1))
    weights = [comp_weight(c, log_p1, rolling, pre_ratio, eve, log_vpc, growth) for c in curves]

    log_q = {tau: [] for tau in LEVELS}
    for h in HOURS:
        weighted = [(c.ratio[h], w) for c, w in zip(curves, weights)]
        plain = [(c.ratio[h], 1.0) for c in curves]
        for tau in LEVELS:
            mixed = (1.0 - POOL_BLEND) * weighted_quantile(weighted, tau) \
                + POOL_BLEND * weighted_quantile(plain, tau)
            log_q[tau].append(mixed)
    if SMOOTH:
        log_q = smooth(log_q)

    rows = []
    for i, h in enumerate(HOURS):
        qs = []
        for tau in LEVELS:
            value = p1 * math.exp(log_q[tau][i])
            qs.append(min(max(value, 0.01), 1e9))
        qs.sort()
        rows.append({"hour": h, "q": [round2(v) for v in qs]})
    return {"hours": rows}


def hour_one_row(case):
    """The case's `known` row for hour 1 (the list's first usable row)."""
    known = case.get("known")
    rows = known if isinstance(known, list) else [known]
    for row in rows:
        if isinstance(row, dict) and row.get("hour") == 1:
            v = row.get("viewers")
            if isinstance(v, (int, float)) and v is not None:
                return row
    for row in rows:
        if isinstance(row, dict):
            v = row.get("viewers")
            if isinstance(v, (int, float)) and v is not None:
                return row
    return None


def hour_one_print(case):
    """Hour 1's viewers from the case's `known` rows."""
    row = hour_one_row(case)
    return float(row["viewers"]) if row is not None else 1.0


def forecast_case(case):
    p1 = hour_one_print(case)
    if not (p1 > 0):
        p1 = 1.0
    try:
        return model_forecast(case, p1)
    except Exception:
        return persistence(p1)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.is_dir():
        return
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(case, dict):
                raise ValueError("case is not an object")
            result = forecast_case(case)
        except Exception:
            result = persistence(1.0)
        text = json.dumps(result, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
