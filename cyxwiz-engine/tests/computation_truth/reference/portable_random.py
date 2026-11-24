"""The Engine's portable seeded draws in Python (cyxwiz/portable_random.h):
std::mt19937 / std::mt19937_64 raw output, the unbiased bounded draw and the
Fisher-Yates shuffle, bit for bit. Shared by the fixture generators that
reproduce an Engine split or shuffle order."""

from __future__ import annotations


class MT19937_64:
    """std::mt19937_64 (the C++ standard's parameters, seed(value))."""
    W = (1 << 64) - 1
    BITS = 64

    def __init__(self, seed):
        self.mt = [0] * 312
        self.mt[0] = seed & self.W
        for i in range(1, 312):
            self.mt[i] = (6364136223846793005 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 62)) + i) & self.W
        self.i = 312

    def __call__(self):
        if self.i >= 312:
            for k in range(312):
                y = (self.mt[k] & 0xFFFFFFFF80000000) | (self.mt[(k + 1) % 312] & 0x7FFFFFFF)
                v = self.mt[(k + 156) % 312] ^ (y >> 1)
                if y & 1:
                    v ^= 0xB5026F5AA96619E9
                self.mt[k] = v
            self.i = 0
        y = self.mt[self.i]
        self.i += 1
        y ^= (y >> 29) & 0x5555555555555555
        y ^= (y << 17) & 0x71D67FFFEDA60000
        y ^= (y << 37) & 0xFFF7EEE000000000
        y ^= y >> 43
        return y & self.W


class MT19937:
    """std::mt19937 (32-bit, seed(value) with the value cut to 32 bits)."""
    W = 0xFFFFFFFF
    BITS = 32

    def __init__(self, seed):
        self.mt = [0] * 624
        self.mt[0] = seed & self.W
        for i in range(1, 624):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & self.W
        self.i = 624

    def __call__(self):
        if self.i >= 624:
            for k in range(624):
                y = (self.mt[k] & 0x80000000) | (self.mt[(k + 1) % 624] & 0x7FFFFFFF)
                v = self.mt[(k + 397) % 624] ^ (y >> 1)
                if y & 1:
                    v ^= 0x9908B0DF
                self.mt[k] = v
            self.i = 0
        y = self.mt[self.i]
        self.i += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & self.W


def bits64(r):
    if getattr(r, "BITS", 64) == 64:
        return r()
    high = r()
    return (high << 32) | r()


def below(r, bound):
    """portable::Below: uniform in [0, bound). A 32-bit engine with a 32-bit
    bound uses Lemire's method with rejection; otherwise 64-bit words above the
    largest multiple of the bound are rejected."""
    if getattr(r, "BITS", 64) == 32 and bound <= 0xFFFFFFFF:
        x = r()
        m = x * bound
        low = m & 0xFFFFFFFF
        if low < bound:
            threshold = (-bound) & 0xFFFFFFFF
            if threshold >= bound:
                threshold -= bound
                if threshold >= bound:
                    threshold %= bound
            while low < threshold:
                x = r()
                m = x * bound
                low = m & 0xFFFFFFFF
        return m >> 32
    mx = (1 << 64) - 1
    limit = mx - mx % bound
    v = bits64(r)
    while v >= limit:
        v = bits64(r)
    return v % bound


def shuffle(values, r):
    """portable::Shuffle: Fisher-Yates from the back, in place."""
    for i in range(len(values), 1, -1):
        j = below(r, i)
        values[i - 1], values[j] = values[j], values[i - 1]
    return values
