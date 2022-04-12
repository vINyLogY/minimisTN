#!/usr/bin/env python
# coding: utf-8
from __future__ import absolute_import, division, print_function

from builtins import filter, map, range, zip
from functools import partial, wraps
from inspect import signature

import numpy as np
from scipy import linalg, integrate, interpolate, fftpack

DTYPE = np.complex128


class GridBasis(object):
    def __init__(self, xmin, xmax, npts, hbar=1.0):
        self.xspace = np.linspace(xmin, xmax, num=npts,
                                  endpoint=True, dtype=DTYPE)
        self.hbar = hbar
        self._conj = None
        return

    def __len__(self):
        return len(self.xspace)

    def __call__(self, f=None):
        return f(self.xspace) if f is not None else self.xspace

    def __getitem__(self, key):
        return self.xspace[key]

    def __eq__(self, other):
        return np.allclose(self(), other())

    @property
    def width(self):
        return self[-1] - self[0]

    def inner_product(self, f1, f2):
        """
        Args:
            f1: float -> float
            f2: float -> float
        Returns:
            float
        """
        v1, v2 = map(self.as_array, (f1, f2))
        return np.sum(integrate.simps(np.conj(v1) * v2, self()))

    def norm(self, f):
        return np.sqrt(self.inner_product(f, f))

    def as_array(self, a):
        return self(a) if callable(a) else np.array(a)

    @staticmethod
    def mop(v):
        """Generate the operator to multiply `v(x)` to the wavefunction
        `psi(x)`.
        Args:
            v: float -> float
        Returns:
            (float -> float) -> (float -> float)
        """
        def operator(f):
            def func(x):
                if np.array(f(x)).ndim <= 1:
                    return v(x) * f(x)
                else:
                    return np.einsum('ijk,jk->ik', v(x), f(x))
            return func
        return operator

    @staticmethod
    def exp(c, v):
        """Generate the operator to multiply `exp(c * v(x))` to the wavefunction
        `psi(x)`, where v(x) is hermite.
        Args:
            c: complex
            v: float -> float
        Returns:
            (float -> float) -> (float -> float)
        """
        def operator(f):
            def func(x):
                if np.array(f(x)).ndim <= 1:
                    ans = np.exp(c * v(x)) * f(x)
                else:
                    a = np.transpose(v(x), (2, 0, 1))
                    b = np.transpose(f(x))
                    # Diagonalization
                    eigenpairs = map(linalg.eigh, a)
                    ans = [np.dot(u, (np.exp(c * w) * np.dot(np.conj(np.transpose(u)), vec)))
                           for (w, u), vec in zip(eigenpairs, b)]
                    ans = np.transpose(ans)
                return ans
            return func
        return operator

    @staticmethod
    def keep_ft_type(func):
        @wraps(func)
        def wrapper(instance, wfn, **kwargs):
            vec = instance.as_array(wfn)
            one_dim = (vec.ndim == 1)
            vec = [vec] if one_dim else list(vec)
            ans = func(instance, vec, **kwargs)
            if one_dim:
                ans = ans[0]
            if callable(wfn):
                ans = interpolate.interp1d(instance.conj(), ans)
            return ans
        return wrapper


class Coordinate(GridBasis):
    def __init__(self, xmin, xmax, npts, hbar=1.0):
        self._conj = None
        super(Coordinate, self).__init__(xmin, xmax, npts, hbar=hbar)
        return

    @property
    def conj(self):
        """Return the conjugate basis in momentum space from coordinate space.
        """
        if self._conj is None:
            width = 2.0 * np.pi * self.hbar * len(self) / (self.width)
            half = 0.5 if len(self) % 2 else 0.5 * (1 - 1.0 / (len(self) - 1))
            co = Momentum((half - 1) * width, half *
                          width, len(self), hbar=self.hbar)
            co.conj = self
            self._conj = co
        return self._conj

    @conj.setter
    def conj(self, value):
        self._conj = value
        return

    @GridBasis.keep_ft_type
    def ft(self, wfn):
        coeff = np.sqrt(2.0 * np.pi * self.hbar) / (self.conj.width)
        ans = [coeff * fftpack.fftshift(fftpack.fft(fftpack.ifftshift(v)))
               for v in wfn]
        return ans


class Momentum(GridBasis):
    def __init__(self, pmin, pmax, npts, hbar=1.0):
        self._conj = None
        super(Momentum, self).__init__(pmin, pmax, npts, hbar=hbar)
        return

    @property
    def conj(self):
        """Return the conjugate basis in coordinate space from momentum space.
        """
        if self._conj is None:
            half_width = np.pi * self.hbar * len(self) / self.width
            co = Coordinate(-half_width, half_width, len(self), hbar=self.hbar)
            co.conj = self
            self._conj = co
        return self._conj

    @conj.setter
    def conj(self, value):
        self._conj = value
        return

    @GridBasis.keep_ft_type
    def ft(self, wfn):
        coeff = np.sqrt(2.0 * np.pi * self.hbar) / self.conj.width * len(self)
        ans = [coeff * fftpack.fftshift(fftpack.ifft(fftpack.ifftshift(v)))
               for v in wfn]
        return ans


def with_time(func, time):
    if 't' in signature(func).parameters:
        func = partial(func, t=time)
    return func


class Propagator(object):
    def __init__(self, init_wfn, basis, func, conj_func):
        assert isinstance(basis, GridBasis)
        assert callable(init_wfn) and callable(func) and callable(conj_func)

        self.wfn = init_wfn
        self.basis = basis
        self.func = func
        self.conj_func = conj_func

        self.n_state = np.size(basis(init_wfn)) // len(basis)
        self.renormalization_coeff = 1.0
        return

    def __call__(self, t0=0, dt=0.1, max_iter=1000, renormalize=False):
        # Assume basis is instace of Coordinate,
        # but codes are identical with Momentum.
        # This method runs on function-level.
        hbar = self.basis.hbar
        fft = self.basis.ft
        ifft = self.basis.conj.ft
        v = self.func
        t = self.conj_func
        exp = self.basis.exp
        yield (t0, self.wfn)
        for i in range(max_iter):
            time = t0 + i * dt
            uv = exp(-1.0j * dt / 2.0 / hbar, with_time(v, time))
            ut = exp(-1.0j * dt / hbar, with_time(t, time))
            self.wfn = uv(ifft(ut(fft(uv(self.wfn)))))
            if renormalize:
                self.normalize()
            yield (time + dt, self.wfn)

    def normalize(self, conj=False):
        basis = self.basis.conj if conj else self.basis
        wfn = self.wfn
        norm = basis.norm(wfn)
        self.renormalization_coeff *= norm
        self.wfn = lambda x: wfn(x) / norm
        return


def gaussian_wfn(x0=0.0, p0=0.0, alpha=1.0, hbar=1.0):
    pi = np.pi
    exp = np.exp

    def gaussian(x):
        return (alpha / pi) ** 0.25 * exp(-alpha * (x-x0)**2 / 2) * exp(1.0j * p0 * (x-x0) / hbar)

    return gaussian


def test_conjconj(npts=1024):
    basis1 = Coordinate(-20.0, 20.0, npts)
    basis2 = basis1.conj.conj
    assert (basis1 == basis2)

    psi1 = gaussian_wfn()
    psi2 = basis1.conj.ft(basis1.ft(psi1))
    assert np.allclose(basis1(psi1), basis2(psi2))
    return


if __name__ == "__main__":
    test_conjconj()
    
