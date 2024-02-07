import random

import dash_chart_editor as dce
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc

from dash import (Dash, Input, Output, State, callback,
                  dcc, html, no_update, register_page, page_container)
from dash.exceptions import PreventUpdate

import plotly.express as px 
import plotly.tools as tls

import openai
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

from environs import Env

import utils

env = Env()
env.read_env('.env')

llm = OpenAI(api_token=env.str('OPENAI_API_KEY'))
openai.OPENAI_API_KEY = env.str('OPENAI_API_KEY')

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_scripts=["https://cdn.plot.ly/plotly-2.18.2.min.js"],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="AI Data Insights",
    use_pages=False,
)

server = app.server


def layout():
    return dmc.MantineProvider(
        [
            utils.jumbotron(),
            page_container,
        ],
    )

app.layout = layout


if __name__ == "__main__":
    app.run(debug=True, port=8056, host='localhost')
