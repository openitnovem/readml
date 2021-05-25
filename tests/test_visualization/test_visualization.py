import os

import plotly.graph_objects as go
from bs4 import BeautifulSoup

from fbd_interpreter.logger import ROOT_DIR
from fbd_interpreter.utils import configuration
from fbd_interpreter.visualization.plots import interpretation_plots_to_html_report

# Get html sections path
html_sections = configuration["PARAMS"]["html_sections"]


def test_interpretation_plots_to_html_report() -> None:
    # Create dummy figure to plot as html with available content of sections
    pdp_dummy_figure = go.FigureWidget()
    pdp_dummy_figure.add_scatter(y=[2, 1, 4, 3])
    dict_figures = {"PDP": pdp_dummy_figure}
    out_path = os.path.join(ROOT_DIR, "../outputs/tests/dummy_figure.html")
    header_pdp = "Dummy figure"
    html = interpretation_plots_to_html_report(
        dict_figures,
        html_sections=html_sections,
        plot_type="PDP",
        title=header_pdp,
        path=out_path,
    )
    # Check if html is valid
    assert bool(BeautifulSoup(html, "html.parser").find())
    # Check if html contains title
    assert header_pdp in html
    # Check if file is created
    assert os.path.exists(out_path)
