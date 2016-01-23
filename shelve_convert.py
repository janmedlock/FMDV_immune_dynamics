#!/usr/bin/python3

import shelve
import os

from herd import parameters


def get_attrs(o):
    return set(x for x in dir(o) if not x.startswith('_'))

def has_current_attrs(p):
    pdefault = parameters.Parameters()
    adefault = get_attrs(pdefault)
    a = get_attrs(p)
    return adefault.issuperset(a)

def get_shelve_key(p):
    # population_size is not in the keys of the current shelves.
    popsize = p.population_size
    del p.population_size
    key = repr(p)
    p.population_size = popsize
    return key
    

shelvepath = 'herd'
shelvefiles = (os.path.join(shelvepath, f)
               for f in os.listdir(shelvepath) if f.endswith('.db'))

for f in shelvefiles:
    print('Converting {}'.format(f))

    sname = f.replace('.db', '')
    with shelve.open(sname) as s:
        for key in s.keys():
            print('key = {}'.format(key))

            p = parameters.Parameters.from_repr(key)
            newkey = get_shelve_key(p)

            if not has_current_attrs(p):
                print('Entry has extinct parameters. Deleting.')
                del s[key]

            elif key != newkey:
                if newkey in s:
                    print('Converted entry is already in cache. Deleting.')
                    del s[key]

                else:
                    print('Replacing entry with converted entry.')
                    s[newkey] = s[key]
                    del s[key]

            else:
                print('Up to date. Doing nothing.')
