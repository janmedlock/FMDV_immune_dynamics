immune_statuses = (
    'maternal_immunity', 'susceptible', 'exposed', 'infectious', 'carrier',
    'recovered_true',       # Apparent     antibodies and true     antibodies
    'recovered_false',      # Apparent     antibodies and true not antibodies
    'lost_immunity_true',   # Apparent not antibodies and true     antibodies
    'lost_immunity_false',  # Apparent not antibodies and true not antibodies
)

# The immune statuses that are infected, i.e. have pathogen
infected = frozenset({'exposed', 'infectious', 'carrier'})
assert infected.issubset(immune_statuses)

# The immune status of mothers that have offspring into 'maternal_immunity'.
# Mothers with all other immune statues have offspring into 'susceptible'.
_offspring_have_maternal_immunity = frozenset({
    'carrier',
    'recovered_true',
    'lost_immunity_false',
})
assert _offspring_have_maternal_immunity.issubset(immune_statuses)

birth_immune_status = {
    immune_status: ('maternal_immunity'
                    if immune_status in _offspring_have_maternal_immunity
                    else 'susceptible')
    for immune_status in immune_statuses
}
