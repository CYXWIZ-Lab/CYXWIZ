# Python plotting examples

Run these in the Engine's Script Editor (open the file, press F5). How each
plot window works is described in `docs/usage/plotting.md`.

| File | What it shows |
| --- | --- |
| `matplotlib_figures.py` | Line, scatter and histogram figures; `plt.show()` puts them in the Plot Output window |
| `rl_metrics.py` | Episode reward/length and policy metrics reported with `pycyxwiz.rl_update_metric`; they appear in the RL Training Dashboard |

The `cyxwiz_plotting` module and its examples were removed (TOFIX134,
decision D3): it could not open Engine windows. A matplotlib-like API that
opens Engine Plot windows replaces it (TOFIX134 P5).
