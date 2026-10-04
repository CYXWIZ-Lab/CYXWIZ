"""matplotlib backend of the Engine (TOFIX134 P5.2): Agg drawing, and show()
sends every open figure to the Engine through cyxwiz_capture, then closes
them. Selected by cyxwiz_capture.install (MPLBACKEND)."""

import io
import sys

from matplotlib._pylab_helpers import Gcf
from matplotlib.backends.backend_agg import _BackendAgg


def _title(fig):
    suptitle = getattr(fig, '_suptitle', None)
    if suptitle is not None and suptitle.get_text():
        return suptitle.get_text()
    for ax in fig.axes:
        if ax.get_title():
            return ax.get_title()
    return ''


def capture_all():
    """Render each open figure to PNG, hand it to the Engine, close them all."""
    capture = sys.modules.get('cyxwiz_capture')
    for manager in Gcf.get_all_fig_managers():
        fig = manager.canvas.figure
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=fig.dpi, bbox_inches='tight')
        width = int(fig.get_figwidth() * fig.dpi)
        height = int(fig.get_figheight() * fig.dpi)
        if capture is not None:
            capture.emit(buf.getvalue(), width, height, _title(fig))
    Gcf.destroy_all()


@_BackendAgg.export
class _BackendCyxWiz(_BackendAgg):
    @staticmethod
    def show(*args, **kwargs):
        capture_all()
