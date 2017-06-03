import numpy as np


# Twiddle algorithm by Phillip J Chase, `Algorithm 382: Combinations of M out of N Objects [G6]',
# Communications of the Association for Computing Machinery 13:6:368 (1970). Code based on
# Matthew Belmonte [http://www.netlib.no/netlib/toms/382].

class Twiddle(object):
    def __init__(self, n, m):
        assert m <= n, "m must be less than or equal to n"
        self.n = n
        self.m = m
        self.p = np.zeros((n + 2), dtype=int)

    def _init_twiddle(self):
        # Initializing Twiddle

        self.p[0] = self.n + 1
        self.p[1:self.n - self.m + 1] = 0
        self.p[self.n - self.m + 1:self.n + 1] = np.arange(1, self.m + 1)
        self.p[self.n + 1] = -2
        if self.m == 0:
            self.p[1] = 1

    def _twiddle(self, x=None, y=None, z=None):

        j = 1
        while self.p[j] <= 0:
            j += 1

        if self.p[j - 1] == 0:
            i = j - 1
            while i != 1:
                self.p[i] = -1
                i -= 1
            self.p[j] = 0
            self.p[1] = 1
            return 0, j - 1, 0, False

        if j > 1:
            self.p[j - 1] = 0
        j += 1
        while self.p[j] > 0:
            j += 1

        k = j - 1
        i = j
        while self.p[i] == 0:
            self.p[i] = -1
            i += 1

        if self.p[i] == -1:
            self.p[i] = self.p[k]
            z = self.p[k] - 1
            self.p[k] = -1
            return i - 1, k - 1, z, False
        elif i != self.p[0]:
            self.p[j] = self.p[i]
            z = self.p[i] - 1
            self.p[i] = 0
            return j - 1, i - 1, z, False
        return x, y, z, True

    def generate(self):
        self._init_twiddle()
        sequences = list()
        combinations = list()

        seq = np.zeros(self.n, dtype=int)
        seq[self.n - self.m:] = 1
        sequences.append(seq.copy())

        comb = np.arange(self.n - self.m, self.n)
        combinations.append(comb.copy())

        x, y, z, done = self._twiddle()
        while not done:
            seq[x] = 1;
            seq[y] = 0;
            sequences.append(seq.copy())

            comb[z] = x
            combinations.append(comb.copy())
            x, y, z, done = self._twiddle(x, y, z)
        return combinations, sequences

    def generate_combinations(self, filepath):
        self._init_twiddle()

        comb = np.arange(self.n - self.m, self.n)

        file=open(filepath,'ab')
        file.write(' '.join(map(str, comb.tolist())) + '\n')
        x, y, z, done = self._twiddle()
        while not done:
            comb[z] = x
            file.write(' '.join(map(str, comb.tolist()))+ '\n')
            x, y, z, done = self._twiddle(x, y, z)
        file.close()
