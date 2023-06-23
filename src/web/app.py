import base64
import io
import json
import os

import dash
import plotly.graph_objects as go

from dash.exceptions import PreventUpdate

from dash import Dash, dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.express as px
import torchvision.transforms as transforms
import torch.nn.functional as F

from PIL import Image

from src.inference import inf_segment
from src.inference.inf_yolo import inference
from src.web.callbacks import get_callbacks
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
get_callbacks(app)
styles = {
    "sidebar": {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "height": "100%",
        "width": "12%",
        "padding": "2% 2%",
        "background-color": "#252525",
        "color": "white",
        "border": "1px solid white",
        "textAlign": "center",
        "display": "flex",
        "flex-direction": "column",
        "justify-content": "space-evenly",
        "align-items": "center"
    },
    "main": {
        "position": "fixed",
        "top": "0%",
        "left": "12%",
        "height": "100%",
        "width": "88%",
        "background-color": "black"
    },
    "header": {
        "position": "relative",
        "top": "2%",
        "left": "2%",
        "right": "2rem",
        "height": "20%",
        "width": "96%",
        "padding": "5 rem 5 rem",
        "background-color": "#353535",
        "color": "white",
        "textAlign": "center",
        "border": "2px solid #fafafa"
    },
    "workzone": {
        "position": "relative",
        "top": "5%",
        "left": "2%",
        "height": "70%",
        "width": "96%",
        "background-color": "#353535",
        "color": "#fafafa"
    },
    "tabs":{ "border": "#454545", "primary": "#676767", "background": "#898989"},
    "upload_field":{
        "border": "1px dashed darkgray",
        "margin": "2.5%",
        "height": "350px",
        "width": "95%",
        "display": "flex", "align-items": "center", "justify-content": "space-around"
    },
    "graph": {
        "position": "relative",
        "left": "1px",
        "width": "99.5%",
        "height": "60%",
        "bottom": "20%",
        "border": "2px solid #fafafa",
        "background-color": "#454545",
        "display": "inline-block"
    },
    "graph_options": {
        "position": "relative",
        "left": "0%",
        "top": "0%",
        "width": "100%",
        "height": "60%",
        "border": "2px solid #fafafa",
        "background-color": "#454545",
        "text-indent": "5%",

        "display": "flex", "flex-direction": "column", "justify-content": "space-evenly", "flex-wrap": "wrap"}
}

header = html.Div(
    [
        html.H1(children='DentalVision', className='header-title'),
        html.P(children='A demo interface for object detection on dental screens', className='header-description'),
    ], className='header', style=styles["header"])

gr_options = html.Div(
            [
                dcc.Checklist(id="graph-options",
                              options=["Labels", "Probabilities"],
                              value=["Labels", "Probabilities"]),
                html.Div(["Line size", dcc.Slider(
                    1, 5, step=1, value=1, id='line-width')]),
            ], style=styles["graph_options"])

sidebar = html.Div(
    [
        html.H2("Sidebar"),
        dbc.Nav(
            [
                #dbc.NavItem(dbc.NavLink("Home", href="", id="page-1-link")),
                dbc.NavItem(dbc.NavLink("Readme", href="https://github.com/Votun/tooth_detection#readme", active="exact"))
            ],  pills=True),
        gr_options
    ], style=styles["sidebar"])

upload_field = dcc.Upload(
    id='upload-img',
    children=html.Div([
        'Drag and Drop or ',
        html.A('Select Files')
    ]), style=styles["upload_field"])

body = html.Div(
    [
        dcc.Tabs(id="tabs-graph", value="raw-img", children=[
            dcc.Tab(label='New', id='new', value="new"),
            dcc.Tab(label='Raw', id='raw-img', value="raw-img"),
            dcc.Tab(label='Teeth Detection', id='detect', value="detect"),
            dcc.Tab(label='Caries Segmentation', id='segment', value="segment"),
        ], colors=styles["tabs"]),
        html.Div(id="output-graph", children=upload_field)
    ], style=styles["workzone"])


app.layout = html.Div([sidebar, html.Div([header, body], style=styles["main"])])

if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0")
