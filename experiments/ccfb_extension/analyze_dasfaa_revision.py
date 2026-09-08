"""Read-only reanalysis of registered trajectories; never fits a model.

This is the authoritative revised fresh-seed analysis. Historical normalized
outputs remain unchanged and must not be substituted for the new raw-gain
results. Bootstrap intervals are descriptive, not multiplicity-adjusted tests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/ccfb_extension/dasfaa_revision'
RAW = ROOT / 'results/ccfb_extension/raw'
FIG = ROOT / 'paper/paper_orchestra/workspace/figures'
BOOT = 10000
RNG_SEED = 20260905
TOL = 1e-10  # Existing numerical tolerance, not selected using outcomes.
METRICS = ['recall', 'brier', 'roc_auc', 'pr_auc', 'fd_ratio']
KEYS = ['dataset', 'seed', 'rho', 'method']
SOURCES: set[Path] = set()


def read(path: Path) -> pd.DataFrame:
    SOURCES.add(path)
    return pd.read_csv(path)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    show = frame.copy().fillna("N/A")
    headers = [str(c) for c in show.columns]
    rows = [[str(value) for value in row] for row in show.itertuples(index=False, name=None)]
    line = lambda values: "| " + " | ".join(values) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *[line(row) for row in rows]])


def load_terminal(experiment: str) -> pd.DataFrame:
    paths = sorted((RAW / experiment).rglob('*.csv'))
    if not paths:
        raise ValueError(f'Missing raw experiment: {experiment}')
    out = pd.concat([read(p).assign(source_path=p.relative_to(ROOT).as_posix()) for p in paths], ignore_index=True)
    out = out.loc[out['round'].eq(10)].copy()
    if out.duplicated(KEYS).any():
        raise ValueError(f'Duplicate terminal cells: {experiment}')
    return out


def recovery(values, noexplore, oracle, higher_better=True):
    """Unnormalized improvement is primary; ratio requires positive oracle gap."""
    a, n, o = np.broadcast_arrays(np.asarray(values, float), np.asarray(noexplore, float), np.asarray(oracle, float))
    direction = 1.0 if higher_better else -1.0
    gain, gap = direction * (a - n), direction * (o - n)
    finite = np.isfinite(a) & np.isfinite(n) & np.isfinite(o)
    flag = np.full(a.shape, 'NONFINITE', dtype=object)
    flag[finite & (gap < 0)] = 'NEGATIVE_GAP'
    flag[finite & (gap == 0)] = 'ZERO_GAP'
    flag[finite & (gap > 0) & (gap <= TOL)] = 'TINY_POSITIVE_GAP'
    valid = finite & (gap > TOL)
    flag[valid] = 'POSITIVE_GAP'
    ratio = np.full(a.shape, np.nan)
    np.divide(gain, gap, out=ratio, where=valid)
    return gain, gap, ratio, flag


def add_references(frame: pd.DataFrame, references: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for method, suffix in [('NoExplore', 'noexplore'), ('Oracle', 'oracle')]:
        ref = references.loc[references.method.eq(method), ['dataset', 'seed'] + METRICS]
        if ref.duplicated(['dataset', 'seed']).any():
            raise ValueError(f'Duplicate references: {method}')
        out = out.merge(ref.rename(columns={m: f'{m}_{suffix}' for m in METRICS}), on=['dataset', 'seed'], how='left', validate='many_to_one')
    for m in METRICS:
        gain, gap, ratio, flag = recovery(out[m], out[f'{m}_noexplore'], out[f'{m}_oracle'], m != 'brier')
        out[f'{m}_gain'], out[f'{m}_oracle_gap'] = gain, gap
        out[f'{m}_recovery_positive_gap'], out[f'{m}_gap_status'] = ratio, flag
    return out


def rank_corr(x, y):
    """Spearman across the final axis, with NaN for constant rank vectors."""
    a, b = rankdata(x, axis=-1), rankdata(y, axis=-1)
    a, b = a - a.mean(axis=-1, keepdims=True), b - b.mean(axis=-1, keepdims=True)
    numerator = (a * b).sum(axis=-1)
    denominator = np.sqrt((a * a).sum(axis=-1) * (b * b).sum(axis=-1))
    return np.divide(numerator, denominator, out=np.full(np.shape(numerator), np.nan), where=denominator > 0)


def cluster_association(frame, xcol, ycol, centered=False, n_boot=BOOT):
    g = frame.loc[np.isfinite(frame[xcol]) & np.isfinite(frame[ycol])].sort_values(['seed', 'rho', 'method']).copy()
    seeds = g.seed.unique()
    parts = [h for _, h in g.groupby('seed', sort=True)]
    if not parts or len({len(h) for h in parts}) != 1:
        raise ValueError('Association requires balanced complete seed clusters')
    x, y = g[xcol].to_numpy(float, copy=True), g[ycol].to_numpy(float, copy=True)
    budget = g.rho.to_numpy(float)
    if centered:
        for r in np.unique(budget):
            mask = budget == r
            x[mask] -= x[mask].mean()
            y[mask] -= y[mask].mean()
    estimate = float(rank_corr(x, y))
    rows_per_seed = len(parts[0])
    rng = np.random.default_rng(RNG_SEED)
    draws = rng.integers(0, len(seeds), (n_boot, len(seeds)))
    indices = (draws[:, :, None] * rows_per_seed + np.arange(rows_per_seed)[None, None, :]).reshape(n_boot, -1)
    # Resample whole seeds: retain every method and budget in each drawn seed.
    bx = g[xcol].to_numpy(float)[indices]
    by = g[ycol].to_numpy(float)[indices]
    if centered:
        br = budget[indices]
        for r in np.unique(budget):
            mask = br == r
            bx -= mask * (np.sum(bx * mask, axis=1) / mask.sum(axis=1))[:, None]
            by -= mask * (np.sum(by * mask, axis=1) / mask.sum(axis=1))[:, None]
    samples = rank_corr(bx, by)
    finite = samples[np.isfinite(samples)]
    lo, hi = np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan)
    return {'n_cells': len(g), 'n_seeds': len(seeds), 'n_methods': g.method.nunique(),
            'spearman_rho': estimate, 'ci_low': lo, 'ci_high': hi,
            'bootstrap_draws': n_boot, 'valid_bootstrap_draws': len(finite),
            'status': 'ESTIMABLE' if np.isfinite(estimate) else 'CONSTANT_INPUT'}


def mean_ci(values, n_boot=BOOT):
    v = np.asarray(values, float)
    if not len(v) or not np.isfinite(v).all():
        raise ValueError('Paired mean requires complete finite values')
    rng = np.random.default_rng(RNG_SEED)
    means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    lo, hi = np.quantile(means, [.025, .975])
    return float(v.mean()), float(lo), float(hi)


def paired_tables(ablation, supplementary):
    main = ablation.loc[np.isclose(ablation.rho, .1)]
    both = pd.concat([main, supplementary], ignore_index=True)
    if both.duplicated(KEYS).any():
        raise ValueError('Duplicate main/supplementary policy cell')
    comparisons = [('LRBE-FAVE', 'CountOnly-MinRisk'), ('LRBE-FAVE', 'QualityFirst-FAVE'),
                   ('RiskBounded-Random', 'CountOnly-MinRisk'), ('RiskBounded-Epsilon', 'CountOnly-MinRisk')]
    metrics = ['cumulative_feedback_count', 'cumulative_predicted_risk', 'recall', 'brier']
    rows = []
    for dataset, g in both.groupby('dataset'):
        for left, right in comparisons:
            a = g.loc[g.method.eq(left)].set_index('seed').sort_index()
            b = g.loc[g.method.eq(right)].set_index('seed').sort_index()
            if not a.index.equals(b.index) or len(a) != 10:
                raise ValueError(f'Incomplete paired seed grid: {dataset}, {left}, {right}')
            if not (a.split_test_hash.equals(b.split_test_hash) and a.split_d0_hash.equals(b.split_d0_hash)):
                raise ValueError('Paired initial split hashes differ')
            for metric in metrics:
                delta = a[metric].to_numpy(float) - b[metric].to_numpy(float)
                mean, lo, hi = mean_ci(delta)
                rows.append(dict(dataset=dataset, left=left, right=right, metric=metric,
                                 n_paired_seeds=len(delta), left_mean=a[metric].mean(), right_mean=b[metric].mean(),
                                 mean_difference=mean, ci_low=lo, ci_high=hi, rho=.1))
    return pd.DataFrame(rows)


def save_latex(paired):
    names = {'hmeq': 'HMEQ', 'lendingclub': 'LendingClub', 'ppdai': 'PPDai', 'uci': 'UCI'}
    metric_cols = ['cumulative_feedback_count', 'cumulative_predicted_risk', 'recall', 'brier']
    lines = [r'\begin{tabular}{llrrrr}', r'\toprule',
             r'Dataset & Contrast & $\Delta K$ & $\Delta\widehat R$ & $\Delta$Recall & $\Delta$Brier \\', r'\midrule']
    labels = {'CountOnly-MinRisk':'MinRisk', 'QualityFirst-FAVE':'QualityFirst', 'LRBE-FAVE':'FAVE',
              'RiskBounded-Random':'Random', 'RiskBounded-Epsilon':'Perturbed'}
    for (dataset, left, right), g in paired.groupby(['dataset', 'left', 'right'], sort=False):
        entries = []
        for metric in metric_cols:
            row = g.loc[g.metric.eq(metric)].iloc[0]
            digits = 1 if metric.startswith('cumulative') else 4
            entries.append(f'{row.mean_difference:.{digits}f} [{row.ci_low:.{digits}f}, {row.ci_high:.{digits}f}]')
        lines.append(' & '.join([names[dataset], f'{labels[left]}--{labels[right]}'] + entries) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (OUT / 'paired_primary_table.tex').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def make_figure(associations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman','DejaVu Serif'],
                         'font.size':8, 'axes.labelsize':8, 'xtick.labelsize':7,
                         'ytick.labelsize':7, 'axes.linewidth':.6, 'savefig.dpi':300})
    q = associations.loc[associations.scope.eq('WITHIN_BUDGET') & associations.rho.eq(.1)]
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    colors = {'recall_gain':'#2060cc', 'brier_gain':'#cc7020'}
    datasets = ['hmeq','lendingclub','ppdai','uci']
    names = ['HMEQ','LendingClub','PPDai','UCI']
    for offset, metric, label in [(-.12,'recall_gain','Recall gain'),(.12,'brier_gain','Brier improvement')]:
        g = q.loc[q.outcome.eq(metric)].set_index('dataset').loc[datasets]
        y = np.arange(4) + offset
        ax.errorbar(g.spearman_rho, y, xerr=np.vstack([g.spearman_rho-g.ci_low,g.ci_high-g.spearman_rho]),
                    fmt='o', ms=3.5, capsize=2, lw=1, color=colors[metric], label=label)
    ax.axvline(0, color='#666666', lw=.7, ls='--')
    ax.set_yticks(range(4), names)
    ax.invert_yaxis()
    ax.set_xlim(-1,1)
    ax.set_xlabel('Spearman correlation with FDRatio gain')
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x', alpha=.15)
    ax.legend(loc='lower left', fontsize=7, frameon=False, bbox_to_anchor=(0,1))
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ['png','pdf']:
        fig.savefig(FIG / f'fig_raw_gain_associations.{ext}', dpi=300)
    plt.close(fig)
    caption = ('Associations between FDRatio gain and downstream improvement relative to NoExplore at '
               'risk-budget proportion 0.10. Points are within-dataset Spearman correlations over 40 '
               'terminal policy cells (four policies, ten seeds); bars are 95% percentile intervals from '
               '10,000 whole-seed bootstrap resamples. Recall gain is policy minus NoExplore; Brier '
               'improvement reverses that subtraction. These descriptive policy-level associations '
               'do not identify a causal effect of feedback composition.')
    (OUT / 'figure_caption.json').write_text(json.dumps({'fig_raw_gain_associations':caption}, indent=2)+'\n', encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    budget, refs = load_terminal('budget_sweep'), load_terminal('reference')
    ablation, supplemental = load_terminal('kstar_ablation'), load_terminal('supplemental_exploration')
    corrected = add_references(budget, refs)
    corrected.to_csv(OUT / 'raw_gains_and_positive_gap_recovery.csv', index=False)
    audit_rows=[]
    unique = corrected.drop_duplicates(['dataset','seed'])
    for dataset, group in unique.groupby('dataset'):
        for metric in METRICS:
            counts = group[f'{metric}_gap_status'].value_counts()
            audit_rows.append({'dataset':dataset, 'metric':metric, 'reference_cells':len(group),
                **{s:int(counts.get(s,0)) for s in ['POSITIVE_GAP','NEGATIVE_GAP','ZERO_GAP','TINY_POSITIVE_GAP','NONFINITE']},
                'minimum_gap':group[f'{metric}_oracle_gap'].min(),'maximum_gap':group[f'{metric}_oracle_gap'].max()})
    audit=pd.DataFrame(audit_rows)
    audit.to_csv(OUT / 'oracle_gap_audit.csv', index=False)
    associations=[]
    for dataset, g in corrected.groupby('dataset'):
        for outcome in ['recall_gain','brier_gain']:
            for rho,h in g.groupby('rho'):
                associations.append(dict(dataset=dataset,rho=rho,outcome=outcome,scope='WITHIN_BUDGET',
                    **cluster_association(h,'fd_ratio_gain',outcome)))
            associations.append(dict(dataset=dataset,rho=np.nan,outcome=outcome,scope='BUDGET_CENTERED',
                **cluster_association(g,'fd_ratio_gain',outcome,centered=True)))
    associations=pd.DataFrame(associations)
    associations.to_csv(OUT / 'raw_gain_associations.csv', index=False)
    paired=paired_tables(ablation,supplemental)
    paired.to_csv(OUT / 'paired_primary_comparisons.csv', index=False)
    save_latex(paired)
    # The archived field is cumulative episode wall time, not isolated policy-selection time.
    runtime=pd.concat([ablation,supplemental],ignore_index=True)
    runtime=runtime.groupby(['experiment','dataset','method'],as_index=False).agg(
        n_runs=('runtime_seconds_elapsed','size'), median_seconds=('runtime_seconds_elapsed','median'),
        min_seconds=('runtime_seconds_elapsed','min'), max_seconds=('runtime_seconds_elapsed','max'))
    runtime.to_csv(OUT / 'descriptive_episode_runtime.csv',index=False)
    make_figure(associations)
    settings={'bootstrap_resamples':BOOT,'bootstrap_seed':RNG_SEED,'positive_gap_tolerance':TOL,
        'primary_gain':'Recall/FDRatio: policy minus NoExplore; Brier: NoExplore minus policy',
        'bootstrap_unit':'seed (all methods and budgets retained); budget means re-estimated each draw',
        'interval':'95% percentile, unadjusted; exploratory descriptive associations',
        'source_experiments':['budget_sweep','reference','kstar_ablation','supplemental_exploration'],
        'frozen_results':'Not pooled with fresh; previous frozen correlations are historical, not revalidated here.',
        'runtime':'Descriptive only. Historical runs are concurrent and selected runtime is end-to-end elapsed, not acquisition-only. Excludes probes, smoke, hardened confirmations, budget sweep duplicates, references.',
        'training_runs_launched':0}
    (OUT / 'analysis_settings.json').write_text(json.dumps(settings,indent=2)+'\n',encoding='utf-8')
    manuscript_stats=associations.loc[associations.scope.eq('BUDGET_CENTERED') | associations.rho.eq(.1)]
    text=['# DASFAA revision: numeric evidence', '',
        'This version replaces negative-gap-normalized fresh-seed mechanism claims with raw improvement analysis. No model was trained; raw and historical derived files were not changed.', '',
        '## Reference-gap audit', markdown_table(audit), '',
        '## Raw-gain associations (base budget and budget-centered secondary)',markdown_table(manuscript_stats),'',
        '## Primary paired comparisons (left minus right; Brier negative is better)',markdown_table(paired),'',
        '## Interpretation limits',
        'All comparisons are policy-level under paired exogenous initial conditions, not shared later candidates/numeric budgets. Similar mean counts and intervals crossing zero are not matching interventions or equivalence tests. Associations do not establish independence or causal composition value. Frozen Phase 4 associations have not been recomputed; do not present them as updated evidence.', '',
        'Positive-gap-only recovery is supplementary and unavailable for negative, zero, tiny-positive (<=1e-10), or nonfinite gaps. Numerical tolerance is inherited, not tuned. Gains remain primary and retain negative-gap cells. FDRatio gain is change in a composition ratio, not a calibrated measure of representativeness; a larger ratio need not be closer to 1.', '',
        'Runtime values are descriptive historical end-to-end wall times affected by concurrent execution. They cannot identify algorithmic overhead. See analysis_settings.json and descriptive_episode_runtime.csv.', '',
        'Reproduce: D:/envs_dirs/no_1/python.exe experiments/ccfb_extension/analyze_dasfaa_revision.py']
    (OUT / 'NUMERIC_EVIDENCE.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
    sources=sorted(SOURCES | {Path(__file__),ROOT/'experiments/ccfb_extension/analyze_exposure_confirmation.py'})
    pd.DataFrame([{'path':p.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in sources]).to_csv(OUT/'source_provenance.csv',index=False)
    print(audit.loc[audit.metric.eq('recall')].to_string(index=False),flush=True)
    print(manuscript_stats.to_string(index=False),flush=True)
    print(f'Wrote {OUT}; {len(SOURCES)} exact source files; zero training runs.',flush=True)


if __name__ == '__main__':
    main()
