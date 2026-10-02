"""Participant-rating protocols, independent of score normalization and fusion."""
import numpy as np
import pandas as pd

PROTOCOLS = ('primary', 'uninstructed')


def apply_rating_protocol(frame, protocol='primary'):
    if protocol not in PROTOCOLS:
        raise ValueError(f'Unknown H–H rating protocol: {protocol}')
    if frame.attrs.get('hh_rating_protocol'):
        raise ValueError('Apply the rating protocol to the original loaded frame only')
    result = frame.copy(deep=True)
    result['mos_eligible'] = True
    result['mos_exclusion_reason'] = ''
    if protocol == 'uninstructed':
        required = {'uninstructed_participant_score', 'uninstructed_combined',
                    'uninstructed_mos_eligible', 'uninstructed_exclusion_reason'}
        if not required.issubset(frame):
            raise ValueError('Uninstructed analysis requires released participant-role columns')
        hh = frame.corpus.isin(('hh_turn', 'hh_emotion'))
        flags = frame.loc[hh, 'uninstructed_mos_eligible']
        if flags.isna().any() or not flags.isin([True, False]).all():
            raise ValueError('Missing or invalid H–H MOS eligibility flags')
        result.loc[hh, 'mos_eligible'] = flags.astype(bool)
        result.loc[hh, 'mos_exclusion_reason'] = frame.loc[hh, 'uninstructed_exclusion_reason'].fillna('')
        excluded = hh & ~result.mos_eligible
        if result.loc[excluded, 'mos_exclusion_reason'].eq('').any():
            raise ValueError('An excluded recording needs an exclusion reason')
        for corpus, dimension in (('hh_turn', 'timing'), ('hh_emotion', 'affect')):
            mask = result.corpus.eq(corpus)
            result.loc[mask, dimension + '_participant'] = frame.loc[mask, 'uninstructed_participant_score']
            result.loc[mask, dimension + '_combined'] = frame.loc[mask, 'uninstructed_combined']
            selected = result.loc[mask & result.mos_eligible]
            p, s, c = (selected[dimension + '_' + view] for view in ('participant', 'supervisor', 'combined'))
            if not np.isfinite(np.column_stack([p, s, c])).all() or not np.allclose(c, (p + s) / 2, atol=1e-12):
                raise ValueError(f'Invalid uninstructed MOS targets: {corpus}')
    result.attrs['hh_rating_protocol'] = protocol
    return result


def mos_cohort(frame):
    return frame.loc[frame.mos_eligible] if 'mos_eligible' in frame else frame


def inter_rater_rows(frame, targets):
    rows = []
    from .metrics import rho
    for corpus, dimension in targets:
        for split in ('dev', 'test', 'all'):
            f = frame.loc[frame.corpus.eq(corpus)]
            if split != 'all':
                f = f.loc[f.split.eq(split)]
            f = mos_cohort(f)
            rows.append(dict(corpus=corpus, dimension=dimension, split=split, n=len(f),
                             sessions=f.group_id.nunique(),
                             value=rho(f[dimension + '_participant'], f[dimension + '_supervisor'])))
    return pd.DataFrame(rows)
