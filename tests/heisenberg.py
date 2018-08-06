#!/usr/bin/env python2
# coding: utf-8
import logging

if __name__ == '__main__':
    import _context

from minitn.dmrg import dmrg1, dmrg2, heisenberg, mat_element, fm_state


def main():
    r"""
    Test program for Heisenberg Model.
    """

    N = 20
    mpo = heisenberg(N)

    mps = dmrg2(mpo, fm_state(N, anti=1), 10, 2)
    mps = dmrg1(mpo, mps, 10, 8)
    energy1 = mat_element(mps, mpo, mps)

    mps = dmrg2(mpo, fm_state(N, anti=1), 10, 10)
    energy2 = mat_element(mps, mpo, mps)

    logging.info('DMRG1 energy: {}'.format(energy1))
    logging.info('DMRG2 energy: {}'.format(energy2))

    return


if __name__ == '__main__':
    main()
