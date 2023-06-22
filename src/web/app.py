import base64
import io
import os

from dash.exceptions import PreventUpdate

from dash import Dash, dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import plotly.express as px

from PIL import Image
from src.inference.inf_yolo import inference

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

styles = {
    "sidebar": {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "height": "100%",
        "width": "10%",
        "padding": "2% 1%",
        "background-color": "#252525",
        "color": "white",
        "border": "1px solid blue",
        "textAlign": "center"
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
                dbc.NavItem(dbc.NavLink("Home", href="", id="page-1-link")),
                dbc.NavItem(dbc.NavLink("Readme", href="/", active="exact")),
            ], pills=True),
        gr_options
    ], style=styles["sidebar"])


workzone = html.Div(
    [
        dcc.Tabs(id="tabs-graph", children=[
            dcc.Tab(label='Raw', id='raw-img'),
            dcc.Tab(label='Teeth Detection', id='detect'),
        ], colors={ "border": '#454545', "primary": '#676767', "background": '#898989',}),
        html.Div(id="output-graph", children=dcc.Upload(
            id='upload-img',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]), style=styles["graph"]),)
    ], style=styles["workzone"])

app.layout = html.Div([sidebar, html.Div([header, workzone], style=styles["main"])])


@callback(Output(component_id="detect", component_property="children"),
          Input(component_id="upload-img", component_property='contents'),
          Input(component_id="graph-options", component_property='value'),
          Input(component_id="line-width", component_property='value'))
def process_img(contents, value, opt_values, line_width):
    if contents is None:
        raise PreventUpdate
    cont = contents.replace("data:image/png;base64,", "")
    msg = base64.b64decode(cont)
    buf = io.BytesIO(msg)
    img = Image.open(buf).convert('RGB')
    if value == "raw":
        fig = px.imshow(img)
    elif value == "detect":
        predict = inference("./", img, 'models/yolo_ext.pt')
        labels = ("Labels" in opt_values)
        probs = ("Probabilities" in opt_values)
        fig = px.imshow(predict.plot(labels=labels, conf=probs, line_width=line_width))
    elif value == "cut":
        fig = px.imshow(img)
    fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, paper_bgcolor="#454545")
    return [dcc.Graph(figure=fig, style=styles["graph"])]


if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0")
