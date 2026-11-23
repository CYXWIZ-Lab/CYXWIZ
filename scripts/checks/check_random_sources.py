"""Seeded draws must repeat on every compiler, OS and device.

Fails when the Engine, backend or server node source uses a random-number call
whose results differ between MSVC, libstdc++ and libc++ (std::shuffle,
std::sample, std::*_distribution, std::default_random_engine, rand) or between
ArrayFire devices (af::randn, engines other than Philox). Use
cyxwiz/portable_random.h and cyxwiz-backend/src/algorithms/device_random.h.

    python check_random_sources.py [<repository root>]
"""
import pathlib
import re
import sys

FOLDERS = ['cyxwiz-engine/src', 'cyxwiz-backend/src', 'cyxwiz-backend/include', 'cyxwiz-server-node/src']
SUFFIXES = {'.cpp', '.h', '.hpp', '.cc', '.cxx', '.inl'}
DISTRIBUTIONS = ('uniform_int|uniform_real|normal|lognormal|exponential|poisson|bernoulli|binomial|negative_binomial|'
                 'geometric|gamma|weibull|extreme_value|chi_squared|cauchy|fisher_f|student_t|discrete|'
                 'piecewise_constant|piecewise_linear')
BANNED = [
    # Bare shuffle( / sample( are left alone: other classes have methods of
    # those names; a using-declaration that brings in std's is caught.
    (re.compile(r'\bstd::(ranges::)?(shuffle|sample|random_shuffle)\s*\(|\busing\s+std::(ranges::)?(shuffle|sample)\b'),
     'std::shuffle / std::sample: use cyxwiz::portable::Shuffle'),
    # With or without std:: (using namespace std, aliases).
    (re.compile(r'\b(' + DISTRIBUTIONS + r')_distribution\b'), 'a std distribution: use the cyxwiz::portable draw'),
    (re.compile(r'\bdefault_random_engine\b'), 'std::default_random_engine: use std::mt19937'),
    (re.compile(r'(?<![\w.>:])(std::|::)?s?rand\s*\('), 'rand(): use std::mt19937 and cyxwiz::portable'),
    (re.compile(r'(?<![\w.>:])(af::)?randn\s*\('), 'af::randn: use cyxwiz::DeviceNormal'),
    (re.compile(r'\bAF_RANDOM_ENGINE_(DEFAULT|THREEFRY\w*|MERSENNE\w*)\b'),
     'an ArrayFire engine named by default or not Philox: use cyxwiz::DeviceRandomEngine'),
]


def strip_comments(text):
    """Source without comments or string literals, line numbers kept."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith('//', i):
            j = text.find('\n', i)
            i = n if j < 0 else j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
            out.append('\n' * text.count('\n', i, j))
            i = j
        elif c in '"\'':
            j = i + 1
            while j < n and text[j] != c and text[j] != '\n':
                j += 2 if text[j] == '\\' else 1
            out.append(c + c)
            i = j + 1
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).resolve().parents[2])
    problems = []
    for folder in FOLDERS:
        base = root / folder
        if not base.is_dir():
            continue
        for path in sorted(base.rglob('*')):
            if path.suffix not in SUFFIXES or not path.is_file():
                continue
            code = strip_comments(path.read_text(encoding='utf-8', errors='replace'))
            for number, line in enumerate(code.split('\n'), 1):
                for pattern, advice in BANNED:
                    if pattern.search(line):
                        problems.append(f'{path.relative_to(root).as_posix()}:{number}: {advice}')
    if problems:
        print('Random draws that differ between platforms or devices:')
        print('\n'.join(problems))
        return 1
    print('random sources: every draw is portable')
    return 0


if __name__ == '__main__':
    sys.exit(main())
