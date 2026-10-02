"""Fill the release table from two verified evaluator runs, not saved table numbers."""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .evaluate import DEFINITIONS, VIEWS
from .presentation import parse_table

METRIC_KEYS = ['scorer', 'corpus', 'dimension', 'split', 'metric', 'rater']
TEST_KEYS = ['corpus', 'dimension', 'metric', 'rater']


def read_run(directory, protocol):
    directory = Path(directory)
    manifest = json.loads((directory / 'run.json').read_text())
    if manifest['hh_rating_protocol'] != protocol:
        raise ValueError(f'Expected {protocol} results in {directory}')
    if manifest['bootstrap'] <= 0:
        raise ValueError('The annotated release table requires bootstrap results')
    for name, digest in manifest['output_sha256'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Result checksum mismatch: {name}')
    tables = {key: pd.read_csv(directory / filename) for key, filename in (
        ('metrics', 'metrics.csv'), ('baselines', 'baseline_comparisons.csv'),
        ('ablations', 'paired_ablations.csv'), ('references', 'inter_rater.csv'))}
    if tables['metrics'][METRIC_KEYS].duplicated().any():
        raise ValueError('Duplicate metric endpoints')
    if not np.isfinite(tables['metrics'].value).all():
        raise ValueError('Undefined metric endpoint')
    for name in ('baselines', 'ablations'):
        t = tables[name]
        if not np.array_equal(t.significant_gain, (t.difference > 0) & (t.p_two_sided < .05)):
            raise ValueError('Significance flags disagree with paired-test results')
    tables['manifest'] = manifest
    tables['manifest_sha256'] = hashlib.sha256((directory / 'run.json').read_bytes()).hexdigest()
    return tables


def validate_pair(primary, sensitivity):
    for key in ('hashes', 'seed', 'bootstrap', 'scoring_configuration', 'code_sha256', 'predictions_sha256'):
        if primary['manifest'][key] != sensitivity['manifest'][key]:
            raise ValueError(f'Evaluation runs differ in {key}')
    if primary['manifest']['output_sha256']['components.csv'] != sensitivity['manifest']['output_sha256']['components.csv']:
        raise ValueError('Rating protocols must not change model components or normalization')
    for key in ('metrics', 'baselines', 'ablations', 'references'):
        a, b = primary[key], sensitivity[key]
        same = a.corpus.eq('hai')
        if 'metric' in a:
            same |= a.metric.isin(('pair_accuracy', 'c_index'))
        cols = [c for c in a.columns if c != 'protocol']
        pd.testing.assert_frame_equal(a.loc[same, cols].reset_index(drop=True),
                                      b.loc[same, cols].reset_index(drop=True))


def table_keys(panel):
    if panel == 'hai':
        return [('hai', d, s, 'rho', v) for d in ('timing', 'affect', 'overall')
                for s in ('dev', 'test') for v in VIEWS]
    pairs = (('hh_turn', 'timing'), ('hh_emotion', 'affect'))
    return ([(c, d, s, m, '-') for c, d in pairs for s in ('dev', 'test')
             for m in ('pair_accuracy', 'c_index')] +
            [(c, d, s, 'rho', v) for c, d in pairs for s in ('dev', 'test') for v in VIEWS])


def rounded(value, metric):
    scale, unit = (100, '.1') if metric == 'pair_accuracy' else (1, '.01')
    return Decimal(str(value * scale)).quantize(Decimal(unit), rounding=ROUND_HALF_UP)


def render_table(template, primary, sensitivity):
    validate_pair(primary, sensitivity)
    panels = parse_table(template)
    runs = {'primary': primary, 'uninstructed': sensitivity}
    lookups = {p: run['metrics'].set_index(METRIC_KEYS) for p, run in runs.items()}
    base = {p: run['baselines'].set_index(TEST_KEYS) for p, run in runs.items()}
    ablations = {p: run['ablations'].set_index([*TEST_KEYS, 'ablation']) for p, run in runs.items()}
    references = primary['references'].set_index(['corpus', 'dimension', 'split'])
    rendered, ledger = [], []
    for panel in panels:
        if panel['kind'] == 'hai' and any('sensitivity' in r['group'] for r in panel['rows']):
            raise ValueError('Use the current template without duplicate H–AI sensitivity rows')
        specs, maxima = [], {}
        for row in panel['rows']:
            name, group = row['name'], row['group']
            protocol = 'uninstructed' if group == 'Uninstructed-participant sensitivity' else 'primary'
            is_ref = name.startswith('Inter-rater')
            values = []
            for i, key in enumerate(table_keys(panel['kind'])):
                c, d, s, metric, view = key
                if is_ref:
                    value = references.loc[(c, d, s), 'value'] if metric == 'rho' and view == 'combined' else None
                elif protocol == 'uninstructed' and metric != 'rho':
                    value = None
                elif name == 'w/o timing outputs' and (panel['kind'] == 'hh' or d != 'overall'):
                    value = None
                else:
                    value = lookups[protocol].loc[(name, *key), 'value']
                values.append(value)
                if value is not None and not is_ref:
                    k = protocol, i
                    maxima[k] = max(maxima.get(k, Decimal('-Infinity')), rounded(value, metric))
            specs.append((row, protocol, is_ref, values))
        for row, protocol, is_ref, values in specs:
            cells = []
            for i, (key, value) in enumerate(zip(table_keys(panel['kind']), values)):
                c, d, s, metric, view = key
                marker = ''
                if value is None:
                    cells.append('--')
                    continue
                number = rounded(value, metric)
                shown = str(abs(number) if number == 0 else number)
                if metric != 'pair_accuracy':
                    shown = shown.replace('-0.', '-.') if shown.startswith('-0.') else shown.removeprefix('0')
                if not is_ref and number == maxima[protocol, i]:
                    shown = r'\textbf{' + shown + '}'
                test_key = c, d, metric, view
                name = row['name']
                if s == 'test' and name == 'Ours' and bool(base[protocol].loc[test_key, 'significant_gain']):
                    marker = r'$^{\dagger}$'
                elif s == 'test' and (*test_key, name) in ablations[protocol].index and bool(ablations[protocol].loc[(*test_key, name), 'significant_gain']):
                    marker = r'$^{\ddagger}$'
                cells.append(shown + marker)
                ledger.append(dict(protocol=protocol, scorer=name, corpus=c, dimension=d, split=s,
                                   metric=metric, rater=view, value=float(value), marker=marker))
            rendered.append((row['name'], '& ' + ' & '.join(cells) + r' \\' + '\n'))
    iterator = iter(rendered)
    lines, pending = [], None
    for line in template.splitlines(keepends=True):
        label = line.strip()
        if not label.startswith('%'):
            if label in DEFINITIONS or label.startswith(r'Inter-rater $\rho$'):
                pending = label
            elif pending and line.startswith('& '):
                name, replacement = next(iterator)
                if name != pending:
                    raise ValueError('Template row order changed')
                line, pending = replacement, None
        lines.append(line)
    if next(iterator, None) is not None:
        raise ValueError('Not all generated rows were inserted')
    text = ''.join(lines)
    # Caption counts and resample count come from the runs, not the template.
    lookup = lookups['uninstructed']
    counts = {(c, s): int(lookup.loc[('Ours', c, d, s, 'rho', 'combined'), 'n'])
              for c, d in (('hh_turn', 'timing'), ('hh_emotion', 'affect')) for s in ('dev', 'test')}
    text, n = re.subn(r'Matched P/S/C MOS counts \(dev/test\) are \d+/\d+ for turn-taking and \d+/\d+ for affect\.',
        f"Matched P/S/C MOS counts (dev/test) are {counts['hh_turn', 'dev']}/{counts['hh_turn', 'test']} for turn-taking and "
        f"{counts['hh_emotion', 'dev']}/{counts['hh_emotion', 'test']} for affect.", text)
    if n != 1:
        raise ValueError('Missing cohort-count caption')
    text, n = re.subn(r'[\d,]+ paired session-bootstrap resamples',
                     f"{primary['manifest']['bootstrap']:,} paired session-bootstrap resamples", text)
    if n != 1:
        raise ValueError('Missing bootstrap caption')
    return text, pd.DataFrame(ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--uninstructed', type=Path, required=True)
    parser.add_argument('--template', type=Path, default=Path('docs/results.tex'))
    parser.add_argument('--output', type=Path, default=Path('docs/results.tex'))
    args = parser.parse_args()
    primary, sensitivity = read_run(args.primary, 'primary'), read_run(args.uninstructed, 'uninstructed')
    template = args.template.read_text()
    text, ledger = render_table(template, primary, sensitivity)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    ledger.to_csv(args.output.with_suffix('.cells.csv'), index=False)
    provenance = dict(
        source='Computed evaluator outputs; template supplies layout and row labels only',
        primary_manifest_sha256=primary['manifest_sha256'],
        uninstructed_manifest_sha256=sensitivity['manifest_sha256'],
        input_sha256=primary['manifest']['hashes'], seed=primary['manifest']['seed'],
        bootstrap=primary['manifest']['bootstrap'],
        scoring_configuration=primary['manifest']['scoring_configuration'],
        per_metric_cohorts={p: run['manifest']['per_metric_cohorts'] for p, run in
                           (('primary', primary), ('uninstructed', sensitivity))},
        reference_protocol='primary', sensitivity_discrimination_display='unchanged; shown as dashes',
        sensitivity_hai_display='unchanged; omitted',
        table_sha256=hashlib.sha256(text.encode()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(f'Generated {args.output}; numeric values and markers recomputed from verified runs.')


if __name__ == '__main__':
    main()
