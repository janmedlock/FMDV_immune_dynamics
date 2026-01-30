'''Handle loading the data.'''

import pathlib

import numpy
import pandas


# The data files are in the same directory as this source file.
_DATA_PATH = pathlib.Path(__file__).parent

# log10 FMDV antibody body titer of at least this value is 'positive',
# below is 'negative'.
ANTIBODY_TITER_CUTOFF = 1.7


def _fix_titer(dfr):
    dfr['titer'] = (
        dfr.loc[:, 'titer']
        .replace({'<1.3': 1.2, '>2.2': 2.3})
        .astype(float)
    )


def _add_positive_negative(dfr):
    dfr['positive'] = dfr['titer'] >= ANTIBODY_TITER_CUTOFF
    dfr['negative'] = ~dfr['positive']


def load_antibodies():
    '''Load the antibody data.'''
    antibodies = pandas.read_csv(_DATA_PATH / 'antibodies.csv')
    _fix_titer(antibodies)
    _add_positive_negative(antibodies)
    return antibodies


def _dtmean(dt, drop_duplicates=False):
    '''Calculate the mean of `pandas.Datetime()`.'''
    if drop_duplicates:
        dt = dt.drop_duplicates()
    if len(dt) == 0:
        return pandas.NaT
    start = dt.iloc[0]
    return (dt - start).mean() + start


def _fix_missing_day(dfr, clean):
    '''Fill in dates that are missing only 'day'.'''
    cols_notnull = ['capture', 'year', 'month']
    ix = (dfr['day'].isnull()
          & dfr[cols_notnull].notnull().all(axis='columns'))
    for (row, sub) in dfr[ix].groupby(cols_notnull):
        row = pandas.Series(row, index=cols_notnull)
        ix_ = clean[cols_notnull].eq(row).all(axis='columns')
        if len(clean[ix_]) > 0:
            val = _dtmean(clean.loc[ix_, 'date'])
        else:
            # There are no other samples with this capture, year, and
            # month. Use the day in this year and month closest to
            # the average date for this capture over all months and
            # years.
            ix_ = clean['capture'] == row['capture']
            mean = _dtmean(clean.loc[ix_, 'date'])
            start = pandas.Timestamp(
                year=row.year, month=row.month, day=1
            )
            end = pandas.offsets.MonthEnd().rollforward(start)
            val = numpy.clip(mean, start, end)
        dfr.loc[sub.index, 'date'] = val


def _fix_missing_year_month_date(dfr, clean):
    '''Fill in dates that missing 'year', 'month', and 'day'.'''
    ix = (dfr[['year', 'month', 'day']].isnull().all(axis='columns')
          & dfr['capture'].notnull())
    for (capture, sub) in dfr[ix].groupby('capture'):
        ix_ = clean['capture'] == capture
        dfr.loc[sub.index, 'date'] = _dtmean(clean.loc[ix_, 'date'])


def _fix_missing_date(dfr):
    clean = dfr.dropna(subset='date')  # Dates not extrapolated.
    _fix_missing_day(dfr, clean)
    _fix_missing_year_month_date(dfr, clean)
    assert dfr['date'].notnull().all()


def _make_date(dfr):
    cols_ymd = ['year', 'month', 'day']
    dfr['date'] = pandas.to_datetime(
        dfr.loc[:, cols_ymd].dropna()
    )
    _fix_missing_date(dfr)
    dfr.drop(columns=cols_ymd, inplace=True)


def load_captures():
    '''Load the capture metadata.'''
    captures = (
        pandas.read_csv(_DATA_PATH / 'captures.csv')
        .astype({col: 'Int64' for col in ('year', 'month', 'day')})
    )
    _make_date(captures)
    return captures


def load():
    '''Load the antibody data with dates.'''
    antibodies = load_antibodies()
    captures = load_captures()
    order = ['ID', 'capture', 'date', 'SAT', 'titer', 'positive', 'negative']
    return (
        antibodies.merge(captures, validate='many_to_one')
        .loc[:, order]
    )


def load_animals():
    '''Load the animal metadata.'''
    return pandas.read_csv(_DATA_PATH / 'animals.csv')


def _get_observations(dfr):
    return (
        # Count of non-null 'titer' values by animal and SAT.
        dfr.groupby(['ID', 'SAT'], observed=True)
        ['titer']
        .count()
        .unstack('SAT')
        .min(axis='columns')  # Take the minimum over SAT.
        .rename('observations')
    )


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


def _get_consecutive_observations(dfr):
    '''Get the length of the longest consecutive non-null subsequence.'''
    lens = (
        dfr
        .set_index(['ID', 'SAT', 'capture'])
        .loc[:, 'titer']
        .sort_index()
        .unstack('capture')
        .agg(_len_consec, axis='columns')
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


def load_observations():
    '''Load animal metadata and observation counts.'''
    animals = (
        load_animals()
        .set_index('ID')
    )
    antibodies = load()
    observations = _get_observations(antibodies)
    consecutive = _get_consecutive_observations(antibodies)
    return pandas.concat([animals, observations, consecutive],
                         axis='columns')
