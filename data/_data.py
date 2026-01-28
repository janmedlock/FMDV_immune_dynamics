'''Handle loading the data.'''

import pathlib

import numpy
import pandas


# The data file is in the same directory as this source file.
DATA_FILE = (
    pathlib.Path(__file__).parent
    / 'Cleaned Data - January 2018.xlsx'
)

# log10 FMDV antibody body titer of at least this value is 'positive',
# below is 'negative'.
ANTIBODY_TITER_CUTOFF = 1.7


def td_to_days(ser):
    '''Convert a `pandas.Timedelta()` to number of days.'''
    return ser / pandas.offsets.Day()


def _tsmean(ts, drop_duplicates=False):
    '''Calculate the mean of a `pandas.Datetime()`.'''
    if drop_duplicates:
        ts = ts.drop_duplicates()
    try:
        return (ts - ts.iloc[0]).mean() + ts.iloc[0]
    except IndexError:
        return pandas.NaT


def _parse_unique_id(ser):
    '''Parse 'Unique ID'.'''
    pat = '(?P<animal>[0-9]+)-(?P<month>[0-9]{2})(?P<year>[0-9]{2})'
    uid = ser.str.extract(pat).astype(int)
    uid.year += 2000
    return uid


def _parse_animal_id(ser):
    '''Parse 'Animal ID'.'''
    # Convert 'C' to '1' and them make them all ints.
    return ser.astype(str).str.replace('C', '1').astype(int)


def _check_id(df):
    '''Check that 'Unique ID', 'Animal ID', and 'Numeric Animal ID' agree.'''
    uid = _parse_unique_id(df['Unique ID'])
    assert df['Numeric Animal ID'].equals(uid.animal)
    assert df['Numeric Animal ID'].equals(_parse_animal_id(df['Animal ID']))


def _load_info():
    info = pandas.read_excel(DATA_FILE,
                             sheet_name='Capture info')
    # Fix data types.
    info = info.astype({col: int
                        for col in ('Numeric Animal ID',
                                    'Capture Number')}
                       | {col: 'Int64'
                          for col in ('Sedation Yr',
                                      'Sedation Month',
                                      'Sedation Day')})
    # Fix some errors.
    info.loc[(info['Unique ID'] == '135-1215'),
             'Animal ID'] = 'C35'
    # This sedation date is wrong, so remove it.
    info.loc[(info['Unique ID'] == '136-0217'),
             ['Sedation Yr', 'Sedation Month', 'Sedation Day']] = numpy.nan
    # Double-check 'Unique ID' etc.
    _check_id(info)
    # Build sample dates.
    dates = info.rename(columns={'Sedation Yr': 'year',
                                 'Sedation Month': 'month',
                                 'Sedation Day': 'day'})
    cols = ['Capture Number', 'year', 'month', 'day']
    dates = dates[cols]
    # Fill in the dates that have no missing values.
    dates['date'] = pandas.to_datetime(dates[cols[1:]].dropna())
    # The dates that have not been extrapolated.
    dates_orig = dates[dates.date.notnull()]
    # Fill in the dates with only day missing.
    cols_null = ['day']
    cols_notnull = list(set(cols) - set(cols_null))
    ix = (dates[cols_null].isnull().all(axis='columns')
          & dates[cols_notnull].notnull().all(axis='columns'))
    for (row, sub) in dates[ix].groupby(cols_notnull):
        row = pandas.Series(row, index=cols_notnull)
        ix_ = dates_orig[cols_notnull].eq(row).all(axis='columns')
        dates_ = dates_orig.date[ix_].dropna()
        if len(dates_) > 0:
            val = _tsmean(dates_)
        else:
            # There are no other samples from this capture, year, and
            # month. Use the day in this year and month closest to
            # the average date for this capture over all months and
            # years.
            ix_ = dates_orig['Capture Number'].eq(row['Capture Number'])
            mean = _tsmean(dates_orig.date[ix_].dropna())
            start = pandas.Timestamp(year=row.year, month=row.month, day=1)
            end = pandas.offsets.MonthEnd().rollforward(start)
            val = numpy.clip(mean, start, end)
        dates.loc[sub.index, 'date'] = val
    # Fill in the dates with only month & day missing.
    cols_null = ['month', 'day']
    cols_notnull = list(set(cols) - set(cols_null))
    ix = (dates[cols_null].isnull().all(axis='columns')
          & dates[cols_notnull].notnull().all(axis='columns'))
    for (row, sub) in dates[ix].groupby(cols_notnull):
        row = pandas.Series(row, index=cols_notnull)
        ix_ = dates_orig[cols_notnull].eq(row).all(axis='columns')
        dates_ = dates_orig.date[ix_].dropna()
        if len(dates_) > 0:
            val = _tsmean(dates_)
        else:
            # There are no other samples from this capture and year.
            # Use the average for this capture over all years.
            ix_ = dates_orig['Capture Number'].eq(row['Capture Number'])
            dates_ = dates_orig.date[ix_].dropna()
            dates.loc[sub.index, 'date'] = _tsmean(dates_)
            # There are no other samples from this capture and
            # year. Use the day in this year closest to the average
            # date for this capture over all years.
            ix_ = dates_orig['Capture Number'].eq(row['Capture Number'])
            mean = _tsmean(dates_orig.date[ix_].dropna())
            start = pandas.Timestamp(year=row.year, month=1, day=1)
            end = pandas.offsets.YearEnd().rollforward(start)
            val = numpy.clip(mean, start, end)
        dates.loc[sub.index, 'date'] = val
    # Fill in the dates with year, month, & day missing.
    cols_null = ['year', 'month', 'day']
    cols_notnull = list(set(cols) - set(cols_null))
    ix = (dates[cols_null].isnull().all(axis='columns')
          & dates[cols_notnull].notnull().all(axis='columns'))
    assert len(cols_notnull) == 1
    for (row, sub) in dates[ix].groupby(cols_notnull[0]):
        row = pandas.Series(row, index=cols_notnull)
        ix_ = dates_orig[cols_notnull].eq(row).all(axis='columns')
        dates_ = dates_orig.date[ix_].dropna()
        dates.loc[sub.index, 'date'] = _tsmean(dates_)
    # Attach dates to info.
    info['date'] = dates.date
    return info


def _load_antibodies():
    antibodies = pandas.read_excel(DATA_FILE,
                                   sheet_name='FMDV Serology')
    # Fix data types.
    antibodies = antibodies.astype({col: int
                                    for col in ('Numeric Animal ID',
                                                'Capture Number')})
    # Fix some errors in 'Unique ID'.
    antibodies.replace({'Unique ID': {'61-0816': '61-0815',
                                      '135-1215': '136-1215'}},
                       inplace=True)
    # Double-check 'Unique ID' etc.
    _check_id(antibodies)
    # Translate upper and lower limits and convert to floats.
    sats = ['SAT-1', 'SAT-2', 'SAT-3']
    with pandas.option_context('future.no_silent_downcasting', True):
        antibodies.replace({sat: {'>2.2': 2.3,
                                  '<1.3': 1.2}
                            for sat in sats},
                           inplace=True)
    return antibodies.astype({col: float
                              for col in sats})


def load():
    '''Load the data.'''
    info = _load_info()
    antibodies = _load_antibodies()
    # Add dates by merging on 'Unique ID' etc.
    cols_merge = ['Unique ID', 'Numeric Animal ID', 'Capture Number', 'date']
    antibodies = pandas.merge(antibodies, info[cols_merge], how='outer')
    # Put SATs into rows.
    antibodies.rename(columns={f'SAT-{x}': f'titer{x}' for x in (1, 2, 3)},
                      inplace=True)
    antibodies.reset_index(inplace=True)
    data = pandas.wide_to_long(antibodies, 'titer', 'Unique ID', 'SAT')
    data.reset_index(inplace=True)
    # Add columns for whether antibodies are positive or negative.
    has_titer = data.titer.notnull()
    data.loc[has_titer, 'positive'] = (
        data.titer[has_titer] >= ANTIBODY_TITER_CUTOFF
    )
    data.loc[has_titer, 'negative'] = (
        data.titer[has_titer] < ANTIBODY_TITER_CUTOFF
    )
    # Drop unwanted columns and reorder.
    cols_keep = cols_merge + ['SAT', 'titer', 'positive', 'negative']
    data = data[cols_keep]
    data.sort_values(['Numeric Animal ID', 'Capture Number', 'SAT'],
                     inplace=True)
    # Index consecutively.
    data.reset_index(drop=True, inplace=True)
    # Fix data types.
    return data.astype({col: 'category'
                        for col in ('Unique ID',
                                    'SAT')}
                       | {col: 'Int64'
                          for col in ('Numeric Animal ID',
                                      'Capture Number')})
