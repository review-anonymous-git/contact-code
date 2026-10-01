from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contact.evaluate import load_inputs, score_column
from contact.metrics import rho
from contact.presentation import parse_table

DATA = Path(__file__).resolve().parents[1] / 'data'


def test_individual_ratings_and_role_mapping():
    f, _ = load_inputs(DATA)
    hh = f.loc[f.corpus.ne('hai')]
    assert len(hh) == 280
    for corpus, dim in [('hh_turn', 'timing'), ('hh_emotion', 'affect')]:
        rows = hh.loc[hh.corpus.eq(corpus)]
        np.testing.assert_allclose(rows[dim + '_participant'],
                                   (rows.participant_1_score + rows.participant_2_score) / 2)
    matched = hh.loc[hh.participant_role_status.eq('unilateral_matched')]
    assert len(matched) == 195
    for n in (1, 2):
        other = 3 - n
        rows = matched.loc[matched[f'participant_{n}_role'].eq('uninstructed')]
        assert rows[f'participant_{other}_role'].eq('instructed').all()
        np.testing.assert_allclose(rows.uninstructed_participant_score, rows[f'participant_{n}_score'])
        np.testing.assert_allclose(rows.instructed_participant_score, rows[f'participant_{other}_score'])


def test_natural_bilateral_and_hai():
    f, _ = load_inputs(DATA)
    natural = f.loc[f.participant_role_status.eq('natural')]
    assert len(natural) == 55
    assert natural.participant_1_role.eq('uninstructed').all()
    assert natural.participant_2_role.eq('uninstructed').all()
    assert natural.instructed_participant_score.isna().all()
    bilateral = f.loc[f.participant_role_status.eq('bilateral')]
    assert len(bilateral) == 30
    assert bilateral.groupby(['corpus', 'split']).size().to_dict() == {
        ('hh_turn', 'dev'): 10, ('hh_turn', 'test'): 20}
    assert bilateral.participant_1_role.eq('instructed').all()
    assert bilateral.participant_2_role.eq('instructed').all()
    assert bilateral.uninstructed_participant_score.isna().all()
    assert bilateral.uninstructed_mos_eligible.eq(False).all()
    hh = f.loc[f.corpus.ne('hai')]
    assert not hh.participant_role_status.str.contains('unresolved').any()
    assert not hh[['participant_1_role', 'participant_2_role']].eq('unresolved').any().any()
    hai = f.loc[f.corpus.eq('hai')]
    assert len(hai) == 213
    assert hai.participant_role_status.eq('not_applicable').all()
    assert hai[['participant_1_score', 'participant_2_score', 'instructed_participant_score',
                'uninstructed_participant_score', 'uninstructed_mos_eligible']].isna().all().all()


def test_uninstructed_table_targets():
    f, _ = load_inputs(DATA)
    eligible = f.loc[f.uninstructed_mos_eligible.eq(True)]
    assert eligible.groupby(['corpus', 'split']).size().to_dict() == {
        ('hh_turn', 'dev'): 50, ('hh_turn', 'test'): 100,
        ('hh_emotion', 'dev'): 32, ('hh_emotion', 'test'): 68}
    for corpus, dim, col, expected in [('hh_turn', 'timing', 'T', .481918),
                                     ('hh_emotion', 'affect', 'E', .282053)]:
        rows = eligible.loc[eligible.corpus.eq(corpus)]
        np.testing.assert_allclose(rows.uninstructed_combined,
                                  (rows.uninstructed_participant_score + rows[dim + '_supervisor']) / 2)
        test = rows.loc[rows.split.eq('test')]
        assert rho(test[col], test.uninstructed_combined) == pytest.approx(expected, abs=5e-7)
    raw = (DATA / 'ratings.csv').read_text()
    import re
    assert re.search(r'\b[0-9a-f]{23,24}\b', raw) is None


@pytest.mark.parametrize('recording,uninstructed_slot,uninstructed,instructed,combined', [
    (9, 1, 4, 4, 3), (12, 2, 4, 2, 3.5), (13, 1, 4, 4, 4), (14, 1, 4, 4, 3),
    (21, 1, 2, 4, 2), (22, 1, 4, 5, 3.5), (23, 2, 4, 2, 3),
    (24, 2, 5, 5, 4), (214, 2, 5, 4, 3.5), (219, 2, 5, 5, 4),
])
def test_confirmed_questionnaire_roles(recording, uninstructed_slot, uninstructed, instructed, combined):
    f, _ = load_inputs(DATA)
    row = f.set_index('recording_id').loc[f'hh_turn__recording-{recording}']
    assert row[f'participant_{uninstructed_slot}_role'] == 'uninstructed'
    assert row[f'participant_{3 - uninstructed_slot}_role'] == 'instructed'
    assert row.uninstructed_participant_score == uninstructed
    assert row.instructed_participant_score == instructed
    assert row.uninstructed_combined == combined
    assert row.participant_role_status == 'unilateral_matched'
    assert row.uninstructed_mos_eligible


@pytest.mark.parametrize('recording,instructed_slot,instructed,uninstructed', [
    (2, 2, 3, 3), (5, 1, 5, 4), (6, 2, 5, 4), (7, 1, 2, 2),
])
def test_session_zero_confirmed_ratings(recording, instructed_slot, instructed, uninstructed):
    f, _ = load_inputs(DATA)
    row = f.set_index('recording_id').loc[f'hh_turn__recording-{recording}']
    assert row[f'participant_{instructed_slot}_role'] == 'instructed'
    assert row.instructed_participant_score == instructed
    assert row.uninstructed_participant_score == uninstructed
    assert row.participant_role_status == 'unilateral_matched'
    assert row.uninstructed_mos_eligible
    assert pd.isna(row.uninstructed_exclusion_reason)


def test_all_hh_exclusions_are_bilateral():
    f, _ = load_inputs(DATA)
    hh = f.loc[f.corpus.ne('hai')]
    excluded = hh.loc[hh.uninstructed_mos_eligible.eq(False)]
    assert len(excluded) == 30
    assert excluded.participant_role_status.eq('bilateral').all()
    assert excluded.uninstructed_exclusion_reason.eq('both_participants_instructed').all()


def test_published_development_correlations_match_eligible_ratings():
    from decimal import Decimal, ROUND_HALF_UP

    f, _ = load_inputs(DATA)
    rows = f.loc[f.corpus.eq('hh_turn') & f.split.eq('dev') & f.uninstructed_mos_eligible.eq(True)]
    panels = parse_table((DATA.parent / 'docs/results.tex').read_text(encoding='utf-8'))
    targets = ['uninstructed_participant_score', 'timing_supervisor', 'uninstructed_combined']
    for entry in panels[0]['rows']:
        if entry['group'] != 'Uninstructed-participant sensitivity':
            continue
        scores = rows[score_column(entry['name'], 'timing')]
        for cell, target in zip(entry['cells'][8:11], targets):
            value = Decimal(str(rho(scores, rows[target]))).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
            assert Decimal(cell['value']) == value
