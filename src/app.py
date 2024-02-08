import random

import os
import base64
import io


import pandas as pd
from dash import Input, Output, State, callback, dcc, html
import dash_ag_grid as dag
import dash_chart_editor as dce
import dash_mantine_components as dmc
import dash_bootstrap_components as dbc

import base64

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

from environs import Env

from pandasai import SmartDataframe
from pandasai.llm import OpenAI

# Set the upload folder path
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'Resources')

env = Env()
env.read_env('.env')

llm = OpenAI(api_token=env.str('OPENAI_API_KEY'))

# print(env.str('OPENAI_API_KEY'))

openai.api_key = env.str('OPENAI_API_KEY')

def get_image_data(file_path):
    image_file = open(file_path, "rb")
    encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string


app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_scripts=["https://cdn.plot.ly/plotly-2.18.2.min.js"],
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="AI Data Insights",
    use_pages=False,
)

app.layout = dbc.Container(
    html.Div(
    [
        dcc.Upload(
            id='upload-file',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select a File')
            ]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            # Allow multiple files to be uploaded
            multiple=False
        ),
        html.Div(id='output-file'),
        dcc.Store(id='output-file-data'),
        html.Div(
            [   
                html.P("Ask about the dataset...", className="lead"),
                dmc.Textarea(
                    placeholder=random.choice(
                        [
                            '"Are there any outliers in this dataset?"',
                            '"What trends do you see in this dataset?"',
                            '"Anything stand out about this dataset?"',
                            '"What columns should I investigate further?"',
                        ]
                    ),
                    autosize=True,
                    minRows=2,
                    id="question",
                    
                ),
                dmc.Group(
                    [
                        dmc.Button(
                            "Submit",
                            id="chat-submit",
                            disabled=True,
                            style={"margin": "10px"}
                        ),
                    ],
                    position="center",
                ),
                dmc.LoadingOverlay(
                    html.Div(
                        id="chat-output",
                        style={"padding": "40px"}
                    ),
                ),
            ],
            id="chat-container",
        ),
        html.Div(
            [   
                html.Div(id='sample_graph'),
                dmc.Textarea(
                    placeholder=random.choice(
                        [
                            '"Do you recommend specific charts given this dataset?"',
                        ]
                    ),
                    autosize=True,
                    minRows=2,
                    id="question-plot",
                    
                ),
                dmc.Group(
                    [
                        dmc.Button(
                            "Generate Plot",
                            id="chat-submit-plot",
                            disabled=True,
                            style={"margin": "10px"}
                        ),
                    ],
                    position="center",
                ),
                dmc.LoadingOverlay(
                    html.Div(
                        id="chat-output-plot",
                        style={"padding": "40px"}
                    ),
                ),
            ],
            id="chat-plot-container",
        ),
        
])
)


def parse_contents(content):
    
    # Decode the file content
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    # Read the file content as a string
    file_content = io.StringIO(decoded.decode('utf-8'))
    return file_content


@app.callback(
    Output('output-file-data', 'data'),
    Output('output-file', 'children'),
    Input('upload-file', 'contents'),
    State('upload-file', 'filename'),
    prevent_initial_call=True)
def update_output(content, name):
    if content is not None:
        data = pd.read_csv(parse_contents(content))
        return data.to_json(date_format='iso', orient='split'), html.H5(f'File Name: {name}')

@callback(
    Output("chat-output", "children"),
    Output("question", "value"),
    Input("chat-submit", "n_clicks"),
    State('output-file-data', 'data'),
    State("question", "value"),
    State("chat-output", "children"),
    prevent_initial_call=True,
)
def chat_window(n_clicks, json_data, question, cur):

    if not json_data:
        raise PreventUpdate
    
    raw_ques = question
    plotly_fig = None

    if json_data is not None:

        df = pd.read_json(json_data, orient='split')
    
        prompt = utils.generate_prompt(df, question)

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}]
        )
        
        question = [
            dcc.Markdown("USER : " +raw_ques, className="chat-item question",style={'color':'blue'}),
            dcc.Markdown("AI : " +
                completion.choices[0].message.content, className="chat-item answer"
            ),
        ]
        
        df = SmartDataframe(df, config={'llm': llm, 'save_charts': True})
        
        #response = df.chat(query=f'Plot {raw_ques}')

        #print("PANDAS AI CHAR RES", response)
        #encoded_string = get_image_data(response)

        #plotly_fig = ''#html.Img(src=f"data:image/png;base64,{encoded_string}", style={'height': '300px'})

        return (question + cur if cur else question), None


@callback(
    Output('sample_graph', 'children'),
    Output("chat-output-plot", "children"),
    Output("question-plot", "value"),
    Input("chat-submit-plot", "n_clicks"),
    State('output-file-data', 'data'),
    State("question-plot", "value"),
    State("chat-output-plot", "children"),
    prevent_initial_call=True,
)
def plot_chat_window(n_clicks, json_data, question, cur):

    if not json_data:
        raise PreventUpdate
    
    raw_ques = question
    plotly_fig = None

    if json_data is not None:

        df = pd.read_json(json_data, orient='split')

        df = SmartDataframe(df, config={'llm': llm, 'save_charts': True})
        
        response = df.chat(query=f'Plot {raw_ques}')

        print("PANDAS AI CHAR RES", response)
        encoded_string = get_image_data(response)

        plotly_fig = html.Img(src=f"data:image/png;base64,{encoded_string}", style={'height': '300px'})

        return plotly_fig, (question + cur if cur else question), None


@callback(Output("chat-submit", "disabled"), Input("question", "value"))
def disable_submit(question):
    return not bool(question)

@callback(Output("chat-submit-plot", "disabled"), Input("question-plot", "value"))
def disable_submit(question):
    return not bool(question)


if __name__ == "__main__":
    app.run(debug=True, port=8056, host='localhost')
