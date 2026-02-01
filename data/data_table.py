#!/usr/bin/python3
'''Handle consecutive observations.'''

import pathlib

import data_


def build_table(observations=None, save=True):
    '''Build the table of observations.'''
    if observations is None:
        observations = data_.load_observations()
    cols_rename = {
        'sex': 'Sex',
        'age_at_first_capture_y': 'Age at first capture (y)',
        'observations': 'Antibody observations',
        'consecutive_observations': 'Consecutive antibody observations',
    }
    table_ = (
        observations
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
    observations_ = data_.load_observations()
    table = build_table(
        observations_,
    )
    print(table)
