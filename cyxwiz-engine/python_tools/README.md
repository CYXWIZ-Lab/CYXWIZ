# python_tools

Python code the Engine runs for its Script Editor's language intelligence
(TOFIX133 P3, decision D2): completion, hover, signature help, go to
definition (Jedi) and problems (pyflakes). The owner chose (2026-10-02) to
ship these with the Engine, so they work offline in every project without
touching the project's environment.

- `cyxwiz_intel.py`: the Engine's service. It imports the packages below
  from this folder without leaving the folder on `sys.path`.
- `wheels/`: the packages as published on PyPI. The build unpacks them next
  to the Engine executable (`<exe dir>/python_tools/`), and release packages
  ship that folder (`redist/scripts/package_release.py`).
- `tests/test_cyxwiz_intel.py`: run by ctest against the unpacked folder.

| Package | Version | Licence | SHA-256 of the wheel |
| --- | --- | --- | --- |
| jedi | 0.20.0 | MIT | 7bdd9c2634f56713299976f4cbd59cb3fa92165cc5e05ea811fb253480728b67 |
| parso | 0.8.7 | MIT | a8926eb2a1b915486941fdbd31e86a4baf88fe8c210f25f2f35ecec5b574ca1c |
| pyflakes | 4.0.1 | MIT | 1559b701962c803eadeb9c215dbac279585367b73ac12e1f0c42868785e6caad |

The licence texts are inside each wheel (`*.dist-info/`) and ship with the
unpacked packages. Jedi needs parso >= 0.8.6, < 0.9.0.

To update: `python -m pip download --no-deps -d wheels jedi parso pyflakes`,
replace the old wheels, update this table, and run the ctest
`python_tools_intel`.
