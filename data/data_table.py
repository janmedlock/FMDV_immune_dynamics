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
    lens_ = (
        dfr
        .set_index(['Numeric Animal ID', 'SAT', 'Capture Number'])
        .loc[:, 'titer']
        .sort_index()
        .unstack('Capture Number')
        .agg(_len_consec, axis='columns')
        .unstack('SAT')
    )
    # Ensure the lengths are the same by SAT.
    assert (
        lens_.min(axis='columns')
        == lens_.max(axis='columns')
    ).all()
    # Just return the first SAT since they're all the same.
    return (
        lens_.iloc[:, 0]
        .rename('Consecutive Observations')
    )


def load_table(save=True):
    '''Build the table of observations.'''
    obs = _data.load_observations()
    consec = load_len_consec()
    cols = [
        'Numeric Animal ID',
        'Sex',
        'Age at First Capture (Years)',
        'Observations',
        'Consecutive Observations',
    ]
    col_index = 'Numeric Animal ID'
    cols_rename = {
        'Numeric Animal ID': 'ID',
        'Age at First Capture (Years)': 'Age at first capture (y)',
        'Observations': 'Antibody observations',
        'Consecutive Observations': 'Consecutive antibody observations',
    }
    replacements = {'Sex': {'M': 'male', 'F': 'female'}}
    table_ = (
        obs.merge(consec, on='Numeric Animal ID')
        .loc[:, cols]
        .rename(columns=cols_rename)
        .replace(replacements)
        .set_index(cols_rename.get(col_index, col_index))
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
    table = load_table()
    print(table)
