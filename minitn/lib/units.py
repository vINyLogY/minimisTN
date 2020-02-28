#!/usr/bin/env python
# coding: utf-8
r"""Dealing with unit transform issues.
Borrowed from ephMPS (https://github.com/jjren/ephMPS/tree/wtli-develop).
"""
from __future__ import division, print_function

import logging

from scipy.constants import physical_constants as constants

# 1 au = `atomic_unit_in['some-unit']` some-unit
atomic_unit_in = {
    None: 1,
    # Energy
    'J': constants["Hartree energy"][0],
    'eV': constants["Hartree energy in eV"][0],
    'meV': 1.e-3 * constants["Hartree energy in eV"][0],
    'cm-1': 0.01 * constants["hartree-inverse meter relationship"][0],
    'K': constants["hartree-kelvin relationship"][0],
    # Time
    's': constants["atomic unit of time"][0],
    'fs': 1.e15 * constants["atomic unit of time"][0],
    'ps': 1.e12 * constants["atomic unit of time"][0],
    'K-1': constants["kelvin-hartree relationship"][0],
    # Length
    'm': constants["atomic unit of length"][0],
    'pm': 1.e12 * constants["atomic unit of length"][0],
    # Mass
    'kg': constants['atomic unit of mass'][0]
}

synonyms = {
    None: ['au', 'a.u.'],
    'K': ['kelvin'],
    'eV': ['ev'],
    'meV': ['mev'],
    'K-1': ['inverse kelvin'],
    'cm-1': ['cm^(-1), cm^{-1}']
}

class Quantity(object):
    def __init__(self, value, unit=None):
        """
        Parameters:
        """
        self.value= float(value)
        self.unit = self.standardize(unit)
        return

    @staticmethod
    def standardize(unit):
        if unit is not None and unit not in atomic_unit_in:
            found = False
            for key, l in synonyms.items():
                if unit in l:
                    unit = key
                    found = True
                    break
            if not found:
                raise ValueError("Cannot recognize unit {} as any in {}."
                                 .format(unit, list(atomic_unit_in.keys())))
        return unit

    @property
    def value_in_au(self):
        return self.value / atomic_unit_in[self.unit]

    def convert_to(self, unit=None):
        unit = self.standardize(unit)
        self.value = self.value_in_au * atomic_unit_in[unit]
        self.unit = unit
        return self

    # a simplified and incomplete model for + - * /
    # + - only allowed between Quantities
    # * / only allowed between Quantities and non-Quantities

    def __neg__(self):
        cls = type(self)
        return cls(-self.value, self.unit)

    def __add__(self, other):
        cls = type(self)
        return cls(self.value_in_au + other.value_in_au)

    def __sub__(self, other):
        cls = type(self)
        return cls(self.value_in_au - other.value_in_au)

    def __mul__(self, other):
        cls = type(self)
        return cls(self.value_in_au * other)

    def __truediv__(self, other):
        cls = type(self)
        return cls(self.value_in_au / other)

    def __eq__(self, other):
        if hasattr(other, "value_in_au"):
            return self.value_in_au == other.value_in_au
        elif other == 0:
            return self.value == 0
        else:
            raise TypeError(
                "Quantity can only compare with Quantity or 0, not {}".format(
                    other.__class__
                )
            )

    def __gt__(self, other):
        if hasattr(other, "value_in_au"):
            return self.value_in_au > other.value_in_au
        elif other == 0:
            return self.value > 0
        else:
            raise TypeError(
                "Quantity can only compare with Quantity or 0, not {}".format(
                    other.__class__
                )
            )

    def __str__(self):
        unit = 'a.u.' if self.unit is None else self.unit
        return "{} {}".format(self.value, unit)

    def __repr__(self):
        return "Quantity<{}>".format(str(self))


if __name__ == '__main__':
    inv_temp = Quantity(1 / 298, unit='K-1').convert_to('au')
    time = Quantity(100, unit='fs').convert_to('au')
    energy = Quantity(2250, unit='cm-1').convert_to(None)
    print(inv_temp, time, energy)
