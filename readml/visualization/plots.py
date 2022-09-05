import base64
from io import BytesIO
from typing import Dict

import plotly.graph_objs as go
import plotly.offline as pyo

from readml.config.sections_html import SECTIONS_HTML


def interpretation_plots_to_html_report(
    dic_figs: Dict[str, go.FigureWidget],
    path: str,
    plot_type: str,
    title: str = "",
):
    """
    Convert a dict of plotly figures to html format.

    :Parameters:
        - dic_figs (Dict[str, go.FigureWidget]):
            Dict of plotly figures to be saved
        - path (str):
            Path to write html file
        - plot_type (str):
            Must be in ["COMMUN", "PDP", "ICE", "ALE", "SHAP]
        - title (str):
            Title of the html report

    :Return:
        string in HTML format
    """
    figs = list(dic_figs.values())
    titles = list(dic_figs.keys())
    dico_sections = SECTIONS_HTML
    add_js = True

    html = """<html><head><meta charset="utf-8"/><style>
            img {
              display: block;
              margin-left: auto;
              margin-right: auto;
            }
            </style></head><body>\n"""
    html += f'<h1 style="color:MediumBlue;text-align:center;font-size:300%">{title}</h1>\n\n'
    html += '<h1 style="color:Navy;font-size:160%">Description générale : </h1>\n\n'

    for element in dico_sections["COMMUN"]:
        html += f'<p style="font-size:120%"> {element} </p>'
    html += "<hr>\n\n"

    html += (
        f'<h1 style="color:Navy;font-size:160%">Description de {plot_type} : </h1>\n\n'
    )
    for element in dico_sections[plot_type]:
        html += f'<p style="font-size:120%"> {element} </p>'
    html += "<hr>\n\n"

    html += (
        '<p style="color:Navy;font-size:160%"> <strong>Features list </strong> : </p>'
    )
    html += "<ul>"
    for feat in titles:
        html += f"<li style=color:Blue><strong><a href=#{feat}> <strong>{feat}</strong></a></li>"
    html += "</ul>"
    html += "<hr>\n\n"

    for idx, fig in enumerate(figs):
        html += f"<section id ={titles[idx]}>"
        html += f'<p style="text-align:center;font-size:160%">{title+" : <strong>"+ titles[idx]+"</strong>"}</p>'
        if plot_type != "SHAP":
            inner_html = pyo.plot(
                fig,
                include_plotlyjs=add_js,
                output_type="div",
            )
        else:
            tmpfile = BytesIO()
            fig.savefig(tmpfile, format="png", bbox_inches="tight")
            encoded = base64.b64encode(tmpfile.getvalue()).decode("utf-8")
            inner_html = f"<img src='data:image/png;base64,{encoded}'>"
        html += inner_html
        html += "</section>"
        html += "<hr>"
        add_js = False

    html += "</body></html>\n"

    with open(path, "w") as resource_file:
        resource_file.write(html)

    return html
