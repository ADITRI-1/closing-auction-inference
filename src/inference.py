import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from .config import (ALPHA, BOUNDARY, MATTERS_CRAMER_V, MATTERS_DELTA_IC, MATTERS_HIT_RATE,
                     MATTERS_VAR_RATIO, N_BOOT, SEED, T2_SECOND, TEST_SECONDS)


def test_sample(df):
    return df[df["seconds_in_bucket"].isin(TEST_SECONDS)].copy()


def smallest_difference(df):
    return float(df["half_spread_bps"].median())


def verdict(ci, low, high=None):
    """'matters' if the whole CI lies beyond the threshold, 'does not matter' if inside it."""
    if high is None:
        if ci[0] > low:
            return "matters"
        if ci[1] < low:
            return "does not matter"
        return "inconclusive"
    if ci[0] > high or ci[1] < low:
        return "matters"
    if ci[0] >= low and ci[1] <= high:
        return "does not matter"
    return "inconclusive"


def t_summary(x, alternative="two-sided", null=0.0):
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = len(x)
    res = stats.ttest_1samp(x, null, alternative=alternative)
    se = x.std(ddof=1) / np.sqrt(n)
    half = stats.t.ppf(1 - ALPHA / 2, n - 1) * se
    return {"n": n, "estimate": float(x.mean()), "se": float(se), "t": float(res.statistic),
            "df": n - 1, "p": float(res.pvalue), "ci": [float(x.mean() - half), float(x.mean() + half)]}


def bootstrap_mean_ci(x, rng):
    x = np.asarray(x, dtype="float64")
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def naive_z(x, alternative="greater"):
    x = np.asarray(x, dtype="float64")
    z = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    p = stats.norm.sf(z) if alternative == "greater" else 2 * stats.norm.sf(abs(z))
    return {"n": len(x), "z": float(z), "p": float(p)}


def test1_direction(s, sdm, full, rng):
    flagged = s[s["imbalance_buy_sell_flag"] != 0]
    edge_w = flagged["imbalance_buy_sell_flag"] * flagged["target_w"]
    edge_raw = flagged["imbalance_buy_sell_flag"] * flagged["target"]
    daily = edge_w.groupby(flagged["date_id"]).mean()
    daily_raw = edge_raw.groupby(flagged["date_id"]).mean()
    primary = t_summary(daily, alternative="greater")
    primary["verdict"] = verdict(primary["ci"], sdm)

    buy = flagged.loc[flagged["imbalance_buy_sell_flag"] == 1, "target_w"]
    sell = flagged.loc[flagged["imbalance_buy_sell_flag"] == -1, "target_w"]
    welch = stats.ttest_ind(buy, sell, equal_var=False, alternative="greater")
    pooled = stats.ttest_ind(buy, sell, equal_var=True, alternative="greater")
    var_f = float(buy.var() / sell.var())
    bf = stats.levene(buy.astype("float64"), sell.astype("float64"), center="median")

    every = full[full["imbalance_buy_sell_flag"] != 0]
    overlapping = naive_z(every["imbalance_buy_sell_flag"] * every["target_w"])

    return {
        "primary": primary,
        "raw_target": t_summary(daily_raw, alternative="greater"),
        "bootstrap_ci": bootstrap_mean_ci(daily.to_numpy(), rng),
        "naive_row_z": naive_z(edge_w),
        "naive_overlapping_z": overlapping,
        "welch_buy_vs_sell": {"t": float(welch.statistic), "p": float(welch.pvalue),
                              "diff": float(buy.mean() - sell.mean()), "n_buy": len(buy), "n_sell": len(sell)},
        "pooled_buy_vs_sell": {"t": float(pooled.statistic), "p": float(pooled.pvalue)},
        "variance_ratio_buy_sell": var_f,
        "brown_forsythe_buy_sell": {"w": float(bf.statistic), "p": float(bf.pvalue)},
        "positive_days": int((daily > 0).sum()),
        "daily": daily.to_numpy().tolist(),
    }


def cramers_v(chi2, n, shape):
    return float(np.sqrt(chi2 / (n * (min(shape) - 1))))


def side_direction_v(frame):
    table = pd.crosstab(frame["imbalance_buy_sell_flag"], frame["target"] > 0)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    return cramers_v(chi2, int(table.to_numpy().sum()), table.shape)


def up_share_gap(frame):
    """Per day: share of up moves after a buy imbalance minus the share after a sell imbalance."""
    return frame.assign(up=frame["target"] > 0).groupby("date_id").apply(
        lambda d: d.loc[d["imbalance_buy_sell_flag"] == 1, "up"].mean()
        - d.loc[d["imbalance_buy_sell_flag"] == -1, "up"].mean(), include_groups=False).dropna()


def bootstrap_v(frame, rng):
    """Percentile interval for Cramer's V, resampling whole days. V >= 0, so the upper end is conservative."""
    _, day = np.unique(frame["date_id"].to_numpy(), return_inverse=True)
    side = frame["imbalance_buy_sell_flag"].to_numpy().astype(int) + 1
    up = (frame["target"] > 0).to_numpy().astype(int)
    counts = np.zeros((day.max() + 1, 3, 2))
    np.add.at(counts, (day, side, up), 1)
    vs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        t = counts[rng.integers(0, len(counts), len(counts))].sum(axis=0)
        t = t[t.sum(axis=1) > 0]
        e = t.sum(axis=1, keepdims=True) * t.sum(axis=0, keepdims=True) / t.sum()
        vs[i] = cramers_v(((t - e) ** 2 / e).sum(), t.sum(), t.shape)
    return [float(np.percentile(vs, 2.5)), float(np.percentile(vs, 97.5))]


def test2_independence(df, s, rng):
    at_second = df["seconds_in_bucket"] == T2_SECOND
    zero_moves = int((at_second & (df["target"] == 0)).sum())
    first = df[at_second & (df["target"] != 0)]
    table =pd.crosstab(first["imbalance_buy_sell_flag"], first["target"] > 0)
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    g, g_p, _, _ = stats.chi2_contingency(table, correction=False, lambda_="log-likelihood")
    n = int(table.to_numpy().sum())
    v = cramers_v(chi2, n, table.shape)

    up = first["target"] > 0
    buy, sell = first["imbalance_buy_sell_flag"] == 1, first["imbalance_buy_sell_flag"] == -1
    p_buy, p_sell = up[buy].mean(), up[sell].mean()
    n_buy, n_sell = int(buy.sum()), int(sell.sum())
    pooled = up[buy | sell].mean()
    z = (p_buy - p_sell) / np.sqrt(pooled * (1 - pooled) * (1 / n_buy + 1 / n_sell))
    se = np.sqrt(p_buy * (1 - p_buy) / n_buy + p_sell * (1 - p_sell) / n_sell)
    diff = p_buy - p_sell
    zc = stats.norm.ppf(1 - ALPHA / 2)

    # With buy/sell and up/down shares near one half, V for buy vs sell equals the up-share gap times this
    # factor (about 1), so the V threshold converts to a threshold on the gap.
    w = n_buy / (n_buy + n_sell)
    v_per_gap = float(np.sqrt(w * (1 - w) / (pooled * (1 - pooled))))
    gap_threshold = MATTERS_CRAMER_V / v_per_gap
    day = t_summary(up_share_gap(first), alternative="greater")

    rows = s[s["target"] != 0]
    all_table = pd.crosstab(rows["imbalance_buy_sell_flag"], rows["target"] > 0)
    all_chi2, all_p, _, _ = stats.chi2_contingency(all_table, correction=False)
    by_minute = {}
    for sec, part in rows.groupby("seconds_in_bucket"):
        gap = t_summary(up_share_gap(part))
        by_minute[int(sec)] = {"v": side_direction_v(part), "gap": gap["estimate"], "gap_ci": gap["ci"]}

    return {
        "primary": {"chi2": float(chi2), "df": int(dof), "p": float(p), "n": n, "cramers_v": v,
                    "min_expected": float(expected.min()), "verdict": verdict(day["ci"], gap_threshold)},
        "zero_moves": zero_moves,
        "v_per_gap": v_per_gap,
        "gap_threshold": float(gap_threshold),
        "bootstrap_v_ci": bootstrap_v(first, rng),
        "phi_buy_sell": side_direction_v(first[buy | sell]),
        "by_minute": by_minute,
        "table": {str(k): {str(c): int(x) for c, x in row.items()} for k, row in table.iterrows()},
        "up_share": {"buy": float(p_buy), "sell": float(p_sell), "none": float(up[first["imbalance_buy_sell_flag"] == 0].mean())},
        "two_proportion": {"z": float(z), "p": float(stats.norm.sf(z)), "diff": float(diff),
                           "ci": [float(diff - zc * se), float(diff + zc * se)]},
        "g_test": {"g": float(g), "p": float(g_p)},
        "day_level": day,
        "all_rows": {"chi2": float(all_chi2), "p": float(all_p), "n": int(all_table.to_numpy().sum()),
                     "cramers_v": cramers_v(all_chi2, int(all_table.to_numpy().sum()), all_table.shape)},
    }


def daily_ic(s):
    ranked = s[["date_id", "post"]].copy()
    keys = [s["date_id"], s["post"]]
    ranked["x"] = s.groupby(keys)["signed_imbalance"].rank()
    ranked["y"] = s.groupby(keys)["target"].rank()
    ic = ranked.groupby(["date_id", "post"]).apply(lambda d: d["x"].corr(d["y"]), include_groups=False)
    return ic.unstack("post").rename(columns={0: "pre", 1: "post"})


def fisher_z(r1, n1, r2, n2):
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return float(z), float(2 * stats.norm.sf(abs(z)))


def corr_t(r, n):
    t = r * np.sqrt((n - 2) / (1 - r**2))
    return {"r": float(r), "n": int(n), "t": float(t), "df": int(n - 2), "p": float(2 * stats.t.sf(abs(t), n - 2))}


def test3_boundary(s):
    ic = daily_ic(s).dropna()
    delta = ic["post"] - ic["pre"]
    primary = t_summary(delta)
    primary["verdict"] = verdict(primary["ci"], -MATTERS_DELTA_IC, MATTERS_DELTA_IC)

    pooled = {}
    for phase, name in [(0, "pre"), (1, "post")]:
        part = s[s["post"] == phase]
        r = stats.spearmanr(part["signed_imbalance"], part["target"]).statistic
        pooled[name] = corr_t(r, len(part))
    fz, fz_p = fisher_z(pooled["post"]["r"], pooled["post"]["n"], pooled["pre"]["r"], pooled["pre"]["n"])
    sign = stats.binomtest(int((delta > 0).sum()), len(delta), 0.5)
    wilcoxon = stats.wilcoxon(delta)

    # Pooling five snapshots could add between-snapshot differences; removing each snapshot's mean rules that out.
    centred = s.copy()
    for c in ["signed_imbalance", "target"]:
        centred[c] = s[c] - s.groupby(["date_id", "seconds_in_bucket"])[c].transform("mean")
    centred_ic = daily_ic(centred).dropna()

    return {
        "primary": primary,
        "ic_pre": t_summary(ic["pre"]),
        "ic_post": t_summary(ic["post"]),
        "pooled_correlation": pooled,
        "fisher_z": {"z": fz, "p": fz_p},
        "sign_test": {"positive_days": int((delta > 0).sum()), "n": len(delta), "p": float(sign.pvalue)},
        "wilcoxon": {"w": float(wilcoxon.statistic), "p": float(wilcoxon.pvalue)},
        "delta_skew": float(stats.skew(delta)),
        "delta_kurtosis": float(stats.kurtosis(delta)),
        "centred": t_summary(centred_ic["post"] - centred_ic["pre"]),
        "daily_pre": ic["pre"].tolist(),
        "daily_post": ic["post"].tolist(),
    }


def f_test(post, pre):
    f = post.var(ddof=1) / pre.var(ddof=1)
    d1, d2 = len(post) - 1, len(pre) - 1
    p = 2 * min(stats.f.cdf(f, d1, d2), stats.f.sf(f, d1, d2))
    ci = [f / stats.f.ppf(1 - ALPHA / 2, d1, d2), f / stats.f.ppf(ALPHA / 2, d1, d2)]
    return {"f": float(f), "df1": d1, "df2": d2, "p": float(p), "ci": [float(c) for c in ci]}


def day_variance_ratio(s, col="target"):
    """t-test on each day's log ratio of post to pre variance, reported on the ratio scale."""
    by_day = s.groupby(["date_id", "post"])[col].var().unstack()
    log_ratio = np.log(by_day[1] / by_day[0]).dropna()
    day = t_summary(log_ratio)
    day["ratio"] = float(np.exp(day["estimate"]))
    day["ratio_ci"] = [float(np.exp(c)) for c in day["ci"]]
    day["days_lower"] = int((log_ratio < 0).sum())
    return day


def test4_variance(s):
    pre = s.loc[s["post"] == 0, "target"].astype("float64")
    post = s.loc[s["post"] == 1, "target"].astype("float64")
    primary = f_test(post, pre)
    bf = stats.levene(post, pre, center="median")

    # Heavy tails and same-day rows make the F interval too narrow, so the verdict uses one ratio per day.
    day = day_variance_ratio(s)
    primary["verdict"] = verdict(day["ratio_ci"], *MATTERS_VAR_RATIO)

    # Exploratory: windows holding an announcement (first imbalance at 15:50, near price at 15:55, the 16:00
    # cross) are the most volatile, so compare with them removed from both sides alike.
    sec = s["seconds_in_bucket"]
    touching = [BOUNDARY - 60, BOUNDARY]
    events = [TEST_SECONDS[0], *touching, TEST_SECONDS[-1]]

    w_pre = s.loc[s["post"] == 0, "target_w"].astype("float64")
    w_post = s.loc[s["post"] == 1, "target_w"].astype("float64")
    return {
        "primary": primary,
        "sd_pre": float(pre.std()), "sd_post": float(post.std()),
        "kurtosis_pre": float(stats.kurtosis(pre)), "kurtosis_post": float(stats.kurtosis(post)),
        "brown_forsythe": {"w": float(bf.statistic), "p": float(bf.pvalue)},
        "day_level_log_ratio": day,
        "winsorised_f": f_test(w_post, w_pre),
        "day_level_clipped": day_variance_ratio(s, "target_w"),
        "without_boundary_windows": day_variance_ratio(s[~sec.isin(touching)]),
        "quiet_minutes_only": day_variance_ratio(s[~sec.isin(events)]),
    }


def test5_hit_rate(s):
    flagged = s[s["imbalance_buy_sell_flag"] != 0]
    gross = flagged["imbalance_buy_sell_flag"] * flagged["target"]
    net = gross - flagged["half_spread_bps"]
    win = (net > 0).to_numpy()
    n, k = len(win), int(win.sum())
    p_hat = k / n
    z = (p_hat - 0.5) / np.sqrt(0.25 / n)
    zc = stats.norm.ppf(1 - ALPHA / 2)
    half = zc * np.sqrt(p_hat * (1 - p_hat) / n)
    ci = [p_hat - half, p_hat + half]
    exact = stats.binomtest(k, n, 0.5, alternative="greater")
    days = flagged["date_id"]
    daily = pd.Series(win, index=flagged.index).groupby(days).mean()

    # A hit rate alone cannot show a loss: check the mean net result per day, and the win and loss sizes.
    net64 = net.astype("float64")
    daily_net = net64.groupby(days).mean()
    mean_win, mean_loss = float(net64[net64 > 0].mean()), float(net64[net64 <= 0].mean())

    # No-information baseline: the same trades and costs with the side chosen by a fair coin.
    y = flagged["target"].astype("float64")
    cost = flagged["half_spread_bps"].astype("float64")
    random_win = 0.5 * (y > cost) + 0.5 * (-y > cost)
    lift = daily - random_win.groupby(days).mean()
    return {
        "primary": {"n": n, "wins": k, "estimate": float(p_hat), "z": float(z), "p": float(stats.norm.sf(z)),
                    "ci": [float(c) for c in ci], "verdict": verdict(ci, MATTERS_HIT_RATE)},
        "gross_hit_rate": float((gross > 0).mean()),
        "mean_net_bps": float(net.mean()),
        "exact_binomial_p": float(exact.pvalue),
        "day_level": t_summary(daily, alternative="greater", null=0.5),
        "daily_net": {**t_summary(daily_net), "positive_days": int((daily_net > 0).sum())},
        "mean_win_bps": mean_win,
        "mean_loss_bps": mean_loss,
        "breakeven_hit_rate": -mean_loss / (mean_win - mean_loss),
        "mean_half_spread_bps": float(cost.mean()),
        "random_side_hit_rate": float(random_win.mean()),
        "lift_over_random": t_summary(lift),
    }


def holm(pvalues):
    names, pvals = list(pvalues), list(pvalues.values())
    reject, adjusted, _, _ = multipletests(pvals, alpha=ALPHA, method="holm")
    return {k: {"p": p, "p_holm": float(a), "reject_holm": bool(r)}
            for k, p, a, r in zip(names, pvals, adjusted, reject)}


def day_level_p(results):
    """Each test's day-level p-value: the primary for Tests 1 and 3, the day-level check for Tests 2, 4 and 5."""
    return {"T1": results["T1"]["primary"]["p"], "T2": results["T2"]["day_level"]["p"],
            "T3": results["T3"]["primary"]["p"], "T4": results["T4"]["day_level_log_ratio"]["p"],
            "T5": results["T5"]["day_level"]["p"]}


def run_tests(df):
    rng = np.random.default_rng(SEED)
    s = test_sample(df)
    sdm = smallest_difference(df)
    results = {
        "T1": test1_direction(s, sdm, df, rng),
        "T2": test2_independence(df, s, rng),
        "T3": test3_boundary(s),
        "T4": test4_variance(s),
        "T5": test5_hit_rate(s),
    }
    for k in results:
        results[k]["primary"]["reject"] = results[k]["primary"]["p"] < ALPHA
    return {"sdm_bps": sdm, "n_test_rows": int(len(s)), "boundary_second": BOUNDARY,
            "tests": results, "holm": holm({k: r["primary"]["p"] for k, r in results.items()}),
            "holm_day": holm(day_level_p(results))}
