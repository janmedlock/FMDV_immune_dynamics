import herd.birth.period


# This function avoids an import loop
# if 'period' is just defined directly.
def get():
    # `herd.birth.gen.hazard` is the only time-dependent function.
    return herd.birth.period.period
