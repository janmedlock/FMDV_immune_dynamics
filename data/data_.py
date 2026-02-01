'''Handle loading the data.'''

import pathlib

import numpy
import pandas


# The data files are in the same directory as this source file.
_DATA_PATH = pathlib.Path(__file__).parent

# log10 FMDV antibody body titer of at least this value is 'positive',
# below is 'negative'.
ANTIBODY_TITER_CUTOFF = 1.7


def load_antibodies():
    '''Load the antibody data.'''

    def fix_titer(antibodies):
        antibodies['titer'] = (
            antibodies.loc[:, 'titer']
            .replace({'<1.3': 1.2, '>2.2': 2.3})
            .astype(float)
        )

    def add_positive_negative(antibodies):
        antibodies['positive'] = antibodies['titer'] >= ANTIBODY_TITER_CUTOFF
        antibodies['negative'] = ~antibodies['positive']

    antibodies = pandas.read_csv(_DATA_PATH / 'antibodies.csv')
    fix_titer(antibodies)
    add_positive_negative(antibodies)
    return antibodies


def load_captures():
    '''Load the capture metadata.'''

    def dtmean(dt, drop_duplicates=False):
        '''Calculate the mean of `pandas.Datetime()`.'''
        if drop_duplicates:
            dt = dt.drop_duplicates()
        if len(dt) == 0:
            return pandas.NaT
        start = dt.iloc[0]
        return (dt - start).mean() + start

    def fix_missing_day(captures, clean):
        '''Fill in dates that are missing only 'day'.'''
        cols_notnull = ['capture', 'year', 'month']
        ix = (captures['day'].isnull()
              & captures[cols_notnull].notnull().all(axis='columns'))
        for (row, sub) in captures[ix].groupby(cols_notnull):
            row = pandas.Series(row, index=cols_notnull)
            ix_ = clean[cols_notnull].eq(row).all(axis='columns')
            if len(clean[ix_]) > 0:
                val = dtmean(clean.loc[ix_, 'date'])
            else:
                # There are no other samples with this capture, year, and
                # month. Use the day in this year and month closest to
                # the average date for this capture over all months and
                # years.
                ix_ = clean['capture'] == row['capture']
                mean = dtmean(clean.loc[ix_, 'date'])
                start = pandas.Timestamp(
                    year=row.year, month=row.month, day=1
                )
                end = pandas.offsets.MonthEnd().rollforward(start)
                val = numpy.clip(mean, start, end)
            captures.loc[sub.index, 'date'] = val

    def fix_missing_year_month_date(captures, clean):
        '''Fill in dates that missing 'year', 'month', and 'day'.'''
        ix = (captures[['year', 'month', 'day']].isnull().all(axis='columns')
              & captures['capture'].notnull())
        for (capture, sub) in captures[ix].groupby('capture'):
            ix_ = clean['capture'] == capture
            captures.loc[sub.index, 'date'] = dtmean(clean.loc[ix_, 'date'])

    def fix_missing_date(captures):
        clean = captures.dropna(subset='date')  # Dates not extrapolated.
        fix_missing_day(captures, clean)
        fix_missing_year_month_date(captures, clean)
        assert captures['date'].notnull().all()

    def make_date(captures):
        cols_ymd = ['year', 'month', 'day']
        captures['date'] = pandas.to_datetime(
            captures.loc[:, cols_ymd].dropna()
        )
        fix_missing_date(captures)
        captures.drop(columns=cols_ymd, inplace=True)

    captures = (
        pandas.read_csv(_DATA_PATH / 'captures.csv')
        .astype({col: 'Int64' for col in ('year', 'month', 'day')})
    )
    make_date(captures)
    return captures


def load_animals():
    '''Load the animal metadata.'''
    return pandas.read_csv(_DATA_PATH / 'animals.csv')


def load(antibodies=None, captures=None):
    '''Load the antibody data with dates.'''
    if antibodies is None:
        antibodies = load_antibodies()
    if captures is None:
        captures = load_captures()
    order = ['ID', 'capture', 'date', 'SAT', 'titer', 'positive', 'negative']
    return (
        antibodies.merge(captures, validate='many_to_one')
        .loc[:, order]
    )


def load_observations(animals=None, data=None,
                      antibodies=None, captures=None):
    '''Load animal metadata and observation counts.'''

    def get_observations(data):
        return (
            # Count of non-null 'titer' values by animal and SAT.
            data.groupby(['ID', 'SAT'], observed=True)
            ['titer']
            .count()
            .unstack('SAT')
            .min(axis='columns')  # Take the minimum over SAT.
            .rename('observations')
        )

    def len_consec(ser):
        '''Get the length of the longest consecutive non-null subsequence.'''
        isnull = ser.isnull()
        if isnull.all():
            return 0
        start = ser.index[~isnull][0]
        if not isnull.loc[start:].any():
            return len(ser.loc[start:])
        end = ser.loc[start:].index[isnull.loc[start:]][0]
        len_ = len(ser.loc[start:end]) - 1
        return max(len_, len_consec(ser.loc[end:]))

    def get_consecutive_observations(data):
        '''Get the length of the longest consecutive non-null subsequence.'''
        lens = (
            data.set_index(['ID', 'SAT', 'capture'])
            .loc[:, 'titer']
            .sort_index()
            .unstack('capture')
            .agg(len_consec, axis='columns')
            .unstack('SAT')
        )
        # Ensure the lengths are the same by SAT.
        assert (
            lens.min(axis='columns')
            == lens.max(axis='columns')
        ).all()
        # Return the first SAT since they're all the same.
        return (
            lens.iloc[:, 0]
            .rename('consecutive_observations')
        )

    if animals is None:
        animals = load_animals()
    if data is None:
        data = load(antibodies=antibodies, captures=captures)
    return pandas.concat(
        [
            animals.set_index('ID'),
            get_observations(data),
            get_consecutive_observations(data),
        ],
        axis='columns',
    )


def load_seropositives(data=None, antibodies=None, captures=None):
    '''Return the seropositives.'''
    if data is None:
        data = load(antibodies=antibodies, captures=captures)
    return data[data['positive']]


def load_with_age(animals=None, captures=None, data=None, antibodies=None):
    '''Load the antibody data with age at each capture.'''

    def get_age(group):
        first = group.loc[group['capture'].idxmin()]
        return (
            (first['age_at_first'] + group['date'] - first['date'])
            / pandas.offsets.Day() / 365
        ).rename('age (y)')

    def load_age(animals=None, captures=None):
        if animals is None:
            animals = load_animals()
        if captures is None:
            captures = load_captures()
        age_at_first = pandas.concat(
            [
                animals['ID'],
                pandas.to_timedelta(
                    animals['age_at_first_capture_y'] * 365,
                    unit='D',
                ).rename('age_at_first'),
            ],
            axis='columns',
        )
        grouper = (
            captures.merge(age_at_first)
            .groupby('ID')
        )
        age = (
            grouper.apply(get_age, include_groups=False)
            .reset_index('ID', drop=True)
        )
        return pandas.concat([captures[['ID', 'capture', 'date']], age],
                             axis='columns')

    if animals is None:
        animals = load_animals()
    if captures is None:
        captures = load_captures()
    age = load_age(animals=animals, captures=captures)
    if data is None:
        data = load(antibodies=antibodies, captures=captures)
    return age.merge(data)
