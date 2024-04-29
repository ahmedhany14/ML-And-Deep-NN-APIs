from collections import Counter, defaultdict, OrderedDict
from queue import Queue, PriorityQueue, LifoQueue, SimpleQueue, SimpleStack


class Hash:
    def __init__(self, s=None, vec=None, Base=0):
        self.n = len(s) if s is not None else len(vec)
        self.p1, self.p2 = 31, 127
        self.m1, self.m2 = 10**9 + 7, 10**9 + 9
        self.pow1, self.pow2, self.h1, self.h2 = (
            [0] * (self.n + 5),
            [0] * (self.n + 5),
            [0] * (self.n + 5),
            [0] * (self.n + 5),
        )

        self.pow1[0] = self.pow2[0] = 1
        for i in range(1, self.n + 1):
            self.pow1[i] = (self.pow1[i - 1] * self.p1) % self.m1
            self.pow2[i] = (self.pow2[i - 1] * self.p2) % self.m2

        if s is not None:
            self.h1[0] = self.h2[0] = 1
            for i in range(1, self.n + 1):
                self.h1[i] = (
                    self.h1[i - 1] * self.p1 + ord(s[i - (not Base)])
                ) % self.m1
                self.h2[i] = (
                    self.h2[i - 1] * self.p2 + ord(s[i - (not Base)])
                ) % self.m2
        elif vec is not None:
            self.h1[0] = self.h2[0] = 1
            for i in range(1, self.n + 1):
                self.h1[i] = (self.h1[i - 1] * self.p1 + vec[i - (not Base)]) % self.m1
                self.h2[i] = (self.h2[i - 1] * self.p2 + vec[i - (not Base)]) % self.m2

    def sub(self, l, r):
        F = self.h1[r]
        F -= self.h1[l - 1] * self.pow1[r - l + 1]
        F = ((F % self.m1) + self.m1) % self.m1

        S = self.h2[r]
        S -= self.h2[l - 1] * self.pow2[r - l + 1]
        S = ((S % self.m2) + self.m2) % self.m2

        return F, S

    def merge_hash(self, l1, r1, l2, r2):
        a = self.sub(l1, r1)
        b = self.sub(l2, r2)
        F = ((a[0] * self.pow1[r2 - l2 + 1]) + b[0]) % self.m1
        S = ((a[1] * self.pow2[r2 - l2 + 1]) + b[1]) % self.m2
        return F, S

    def at(self, idx):
        return self.sub(idx, idx)

    def equal(self, l1, r1, l2, r2):
        return self.sub(l1, r1) == self.sub(l2, r2)

