# coding: utf-8
"""Adopt from OpenQuant (https://github.com/hou-dao/OpenQuant.git)
"""

from minitn.lib.backend import np


def tseig(D, E):
    mat = np.diag(E, -1) + np.diag(D, 0) + np.diag(E, 1)
    return -np.sort(-np.linalg.eigvalsh(mat))


def MSD(N, BoseFermi=1):
    assert N >= 0 and BoseFermi in {1, 2}

    if BoseFermi == 1:
        pole = np.array([2 * (i + 1) * np.pi for i in range(N)])
        resi = np.ones(N, dtype=float)
        return pole, resi, 0.0, 0.0
    elif BoseFermi == 2:
        pole = np.array([(2 * i + 1) * np.pi for i in range(N)])
        resi = np.ones(N, dtype=float)
        return pole, resi, 0.0, 0.0


def PSD(N, BoseFermi=1, pade=2):
    assert N >= 0 and BoseFermi in {1, 2} and pade in {0, 1, 2, 3}

    if pade == 0:
        return MSD(N, BoseFermi)

    elif pade == 1 or pade == 2:
        pole, resi = [], []
        if N > 0:
            M = 2 * N + pade // 2
            temp = 3.0 if BoseFermi == 1 else 1.0
            diag = np.zeros(M, dtype=float)
            doff = np.array([1.0 / np.sqrt((temp + 2.0 * i) * (temp + 2.0 * (i + 1))) for i in range(M - 1)])
            pole = 2.0 / tseig(diag, doff)[:N]
            pol2 = np.array([x * x for x in pole])
            M -= 1
            temp = 5.0 if BoseFermi == 1 else 3.0
            diag = np.zeros(M, dtype=float)
            doff = np.array([1.0 / np.sqrt((temp + 2.0 * i) * (temp + 2.0 * (i + 1))) for i in range(M - 1)])
            M //= 2
            eig2 = np.power(2.0 / tseig(diag, doff)[:M], 2)
            scaling = 0.0
            if BoseFermi == 1:
                scaling = N * (2.0 * N + 3.0) if pade == 1 else 1.0 / (4.0 * (N + 1.0) * (2.0 * N + 3.0))
            elif BoseFermi == 2:
                scaling = N * (2.0 * N + 1.0) if pade == 1 else 1.0 / (4.0 * (N + 1.0) * (2.0 * N + 1.0))
            resi = np.zeros(N, dtype=float)
            for j in range(N):
                if pade == 2:
                    temp = 0.5 * scaling * (eig2[j] - pol2[j])
                elif pade == 1:
                    if j == N - 1:
                        temp = 0.5 * scaling
                    else:
                        temp = 0.5 * scaling * (eig2[j] - pol2[j]) / (pol2[N - 1] - pol2[j])
                for k in range(M):
                    temp *= (eig2[k] - pol2[j]) / (pol2[k] - pol2[j]) if k != j else 1.0
                resi[j] = temp
        rn, tn = 0.0, 0.0
        if BoseFermi == 1 and pade == 2:
            rn = 1.0 / (4.0 * (N + 1.0) * (2.0 * N + 3.0))
        return pole, resi, rn, tn

    elif pade == 3:
        Np1 = N + 1
        temp = 3.0 if BoseFermi == 1 else 1.0
        d = np.empty(2 * Np1, dtype=float)
        d[0] = 0.25 / temp
        d[-1] = -4.0 * (N + 1.0) * (N + 1.0) * (temp + 2 * N) * (temp + 2 * N) * (temp + 4 * N + 2.0)
        for i in range(1, Np1):
            d[2 * i - 1] = -4.0 * i * i * (temp + 2.0 * i - 2.0) * (temp + 2.0 * i - 2.0) * (temp + 4.0 * i - 2.0)
            d[2 * i] = -0.25 * (temp + 4.0 * i) / i / (i + 1) / (temp + 2.0 * i - 2.0) / (temp + 2.0 * i)
        sumd2 = np.empty(Np1, dtype=float)
        sumd2[0] = d[1]
        for i in range(1, Np1):
            sumd2[i] = sumd2[i - 1] + d[2 * i + 1]
        tn = 0.25 / sumd2[-1]
        rn = sum(d[2 * i] * (4.0 * tn * (sumd2[-1] - sumd2[i - 1]))**2 if i > 0 else d[2 * i] for i in range(Np1))
        M = 2 * N + 1
        diag = np.zeros(M, dtype=float)
        doff = np.array([1.0 / np.sqrt(d[i + 1] * d[i + 2]) for i in range(M - 1)])
        pole = 2.0 / tseig(diag, doff)[:N]
        resi = np.zeros(N, dtype=float)
        for j in range(N):
            scaling = pole[j] * pole[j]
            r0, t1 = 0.0, 0.25 / d[1]
            eta0, eta1, eta2 = 0.0, 0.5, 0.0
            for i in range(Np1):
                r1 = t1 if (i == j or i == N) else t1 / (pole[i] * pole[i] - scaling)
                r2 = 2.0 * np.sqrt(abs(r1)) if r1 > 0 else -2.0 * np.sqrt(abs(r1))
                r1 = 2.0 * np.sqrt(abs(r1))
                eta2 = d[2 * i] * r1 * eta1 - 0.25 * r1 * r0 * scaling * eta0
                eta0 = eta1
                eta1 = eta2
                eta2 = d[2 * i + 1] * r2 * eta1 - 0.25 * r2 * r1 * scaling * eta0
                eta0 = eta1
                eta1 = eta2
                r0 = r2
                if i != N:
                    t1 = sumd2[i] / sumd2[i + 1]
            resi[j] = eta2
        return pole, resi, rn, tn
