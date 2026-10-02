import copy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contact.evaluate import load_inputs, metric_rows, run_evaluation
from contact.presentation import parse_table, render_document
from contact.protocols import apply_rating_protocol
from contact.report import read_run, render_table, validate_pair

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def frame():
    return load_inputs(ROOT / 'data')[0]


@pytest.fixture(scope='module')
def evaluated(tmp_path_factory):
    root = tmp_path_factory.mktemp('protocols')
    for protocol in ('primary', 'uninstructed'):
        run_evaluation(ROOT / 'data', root / protocol, protocol, bootstrap=63, seed=20260925)
    return root


def test_default_targets_and_scores_unchanged(frame):
    original = frame.copy(deep=True)
    primary = apply_rating_protocol(frame)
    pd.testing.assert_frame_equal(primary[frame.columns], original, check_flags=False)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(metric_rows(primary), metric_rows(frame))


def test_uninstructed_cohort_does_not_filter_discrimination(frame):
    sensitivity = apply_rating_protocol(frame, 'uninstructed')
    assert len(sensitivity) == 493
    assert sensitivity.loc[sensitivity.mos_eligible].groupby(['corpus', 'split']).size().to_dict() == {
        ('hh_turn', 'dev'): 50, ('hh_turn', 'test'): 100,
        ('hh_emotion', 'dev'): 32, ('hh_emotion', 'test'): 68,
        ('hai', 'dev'): 70, ('hai', 'test'): 143}
    score_columns = ['recording_id', 'F_raw', 'S_raw', 'A_raw', 'F', 'S', 'A', 'T', 'E', 'O']
    pd.testing.assert_frame_equal(frame[score_columns], sensitivity[score_columns])
    a, b = metric_rows(frame), metric_rows(sensitivity)
    unchanged = a.metric.ne('rho') | a.corpus.eq('hai')
    pd.testing.assert_frame_equal(a.loc[unchanged].drop(columns='protocol'),
                                  b.loc[unchanged].drop(columns='protocol'))
    tt = b.loc[b.scorer.eq('Ours') & b.corpus.eq('hh_turn') & b.split.eq('test')]
    assert set(tt.loc[tt.metric.eq('rho'), 'n']) == {100}
    assert set(tt.loc[tt.metric.ne('rho'), 'n']) == {120}
    assert tt.loc[tt.rater.eq('combined'), 'value'].item() == pytest.approx(.4819178715)
    supervisors = [c for c in frame if c.endswith('_supervisor')]
    pd.testing.assert_frame_equal(frame[supervisors], sensitivity[supervisors])


def test_invalid_or_reapplied_protocol_rejected(frame):
    with pytest.raises(ValueError, match='Unknown'):
        apply_rating_protocol(frame, 'typo')
    with pytest.raises(ValueError, match='original loaded'):
        apply_rating_protocol(apply_rating_protocol(frame), 'uninstructed')
    with pytest.raises(ValueError, match='requires'):
        apply_rating_protocol(frame.drop(columns='uninstructed_mos_eligible'), 'uninstructed')
    broken = frame.copy()
    broken.loc[broken.corpus.eq('hh_turn'), 'uninstructed_combined'] = 999
    with pytest.raises(ValueError, match='Invalid uninstructed'):
        apply_rating_protocol(broken, 'uninstructed')


def test_manifests_and_paired_test_populations(evaluated):
    a, b = read_run(evaluated / 'primary', 'primary'), read_run(evaluated / 'uninstructed', 'uninstructed')
    validate_pair(a, b)
    manifest = b['manifest']
    assert len(manifest['mos_exclusions']) == 30
    assert {row['mos_exclusion_reason'] for row in manifest['mos_exclusions']} == {'both_participants_instructed'}
    assert manifest['scoring_configuration']['timing_activity_weight'] == .45
    assert manifest['scoring_configuration']['affect_av_weight'] == .25
    assert manifest['hashes']['ratings.csv'] == hashlib.sha256((ROOT / 'data/ratings.csv').read_bytes()).hexdigest()
    assert 'not a new blind test' in manifest['status']
    assert 'speaker-disjoint evaluation split' in manifest['status']
    tests = b['baselines'].loc[b['baselines'].corpus.eq('hh_turn')]
    assert set(tests.loc[tests.metric.eq('rho'), 'n']) == {100}
    assert set(tests.loc[tests.metric.ne('rho'), 'n']) == {120}
    assert set(tests.inventory_n) == {120}
    assert set(tests.sessions) == {20}
    with pytest.raises(FileExistsError):
        run_evaluation(ROOT / 'data', evaluated / 'primary', 'uninstructed')
    altered = copy.deepcopy(b)
    altered['manifest']['seed'] += 1
    with pytest.raises(ValueError, match='seed'):
        validate_pair(a, altered)


def test_display_uses_computed_values_and_markers(evaluated):
    a, b = read_run(evaluated / 'primary', 'primary'), read_run(evaluated / 'uninstructed', 'uninstructed')
    template = (ROOT / 'docs/results.tex').read_text()
    text, ledger = render_table(template, a, b)
    corrupted = ''.join(line.replace('& 44.0 &', '& 99.9 &').replace(r'$^{\dagger}$', r'$^{\ddagger}$')
                        if line.startswith('& ') else line for line in template.splitlines(keepends=True))
    rebuilt, _ = render_table(corrupted, a, b)
    assert rebuilt == text
    panels = parse_table(text)
    assert '63 paired session-bootstrap' in render_document(panels)
    sensitivity = [r for r in panels[0]['rows'] if r['group'] == 'Uninstructed-participant sensitivity']
    assert len(sensitivity) == 4
    assert all(c['value'] == '--' and c['marker'] is None for r in sensitivity for c in r['cells'][:8])
    assert not any('sensitivity' in r['group'] for r in panels[1]['rows'])
    assert '63 paired session-bootstrap resamples' in text
    assert '50/100 for turn-taking' in text
    for protocol, run in (('primary', a), ('uninstructed', b)):
        marks = ledger.loc[ledger.protocol.eq(protocol) & ledger.scorer.eq('Ours') & ledger.split.eq('test')]
        for _, row in marks.iterrows():
            test = run['baselines']
            test = test.loc[test.corpus.eq(row.corpus) & test.dimension.eq(row.dimension)
                            & test.metric.eq(row.metric) & test.rater.eq(row.rater)].iloc[0]
            assert bool(row.marker) == bool(test.significant_gain)


def test_result_tampering_rejected(evaluated, tmp_path):
    import shutil
    shutil.copytree(evaluated / 'primary', tmp_path / 'primary')
    metrics = tmp_path / 'primary/metrics.csv'
    metrics.write_text(metrics.read_text() + '\n')
    with pytest.raises(ValueError, match='checksum'):
        read_run(tmp_path / 'primary', 'primary')
