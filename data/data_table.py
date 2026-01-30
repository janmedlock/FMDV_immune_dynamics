#!/usr/bin/python3
'''Handle consecutive observations.'''

import pathlib

import _data


def _len_consec(ser):
    '''Get the length of the longest consecutive non-null subsequence.'''
    isnull = ser.isnull()
    if isnull.all():
        return 0
    start = ser.index[~isnull][0]
    if not isnull.loc[start:].any():
        return len(ser.loc[start:])
    end = ser.loc[start:].index[isnull.loc[start:]][0]
    len_ = len(ser.loc[start:end]) - 1
    return max(len_, _len_consec(ser.loc[end:]))


def load_len_consec():
    '''Get the length of the longest consecutive non-null subsequence.'''
    dfr = _data.load()
    return _data.consecutive_observations(dfr)


def build_table(save=True):
    '''Build the table of observations.'''
    cols_rename = {
        'sex': 'Sex',
        'age_at_first_capture_y': 'Age at first capture (y)',
        'observations': 'Antibody observations',
        'consecutive_observations': 'Consecutive antibody observations',
    }
    table_ = (
        _data.load_observations()
        .rename(columns=cols_rename)
    )
    if save:
        path = pathlib.Path(__file__).with_suffix('.tex')
        styler = (
            table_.style
            .format(
                precision=1,
            )
            .format_index(
                lambda x: rf'\multicolumn{{1}}{{l}}{{\textbf{{{x}}}}}',
                axis='columns',
            )
        )
        styler.to_latex(
            path,
            position='!h',
            environment='longtable',
            hrules=True,
            label='table:data_summary',
            caption='Data summary.',
        )
    return table_


if __name__ == '__main__':
    table = build_table()
    print(table)
