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
styles = {
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
}
def get_callbacks(app):
    @callback(Output(component_id="detect", component_property="children"),
              Input(component_id="upload-img", component_property='contents'),
              Input(component_id="graph-options", component_property='value'),
              Input(component_id="line-width", component_property='value'))
    def process_img(contents, opt_values, line_width):
        if contents is None:
            raise PreventUpdate
        cont = contents.replace("data:image/png;base64,", "")
        msg = base64.b64decode(cont)
        buf = io.BytesIO(msg)
        img = Image.open(buf).convert('RGB')
        predict = inference("./", img, 'models/yolo_ext.pt')
        labels = ("Labels" in opt_values)
        probs = ("Probabilities" in opt_values)
        fig = px.imshow(predict.plot(labels=labels, conf=probs, line_width=line_width))
        fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, paper_bgcolor="#454545")
        return [dcc.Graph(figure=fig, style=styles["graph"])]

    @callback(Output(component_id="segment", component_property="children"),
              Input(component_id="upload-img", component_property='contents'))
    def process_img(contents):
        if contents is None:
            raise PreventUpdate
        cont = contents.replace("data:image/png;base64,", "")
        toPIL = transforms.ToPILImage()
        msg = base64.b64decode(cont)
        buf = io.BytesIO(msg)
        img = Image.open(buf).convert('RGB')
        predict = inference("./", img, 'models/yolo_ext.pt')
        mask = inf_segment.inf_seg(predict)
        fig = px.imshow(img)
        fig.add_layout_image(
            dict(
                source=toPIL(mask),
                xref="x domain",
                yref="y domain",
                x=1,
                y=1,
                xanchor="right",
                yanchor="top",
                opacity=0.3,
                sizex=1,
                sizey=1)
        )
        fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, paper_bgcolor="#454545")
        return [dcc.Graph(figure=fig, style=styles["graph"])]

    @callback(Output(component_id="raw-img", component_property="children"),
              Input(component_id="upload-img", component_property='contents')
              )
    def detect_img(contents):
        if contents is None:
            raise PreventUpdate
        cont = contents.replace("data:image/png;base64,", "")
        msg = base64.b64decode(cont)
        buf = io.BytesIO(msg)
        img = Image.open(buf).convert('RGB')
        fig = px.imshow(img)
        fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, paper_bgcolor="#454545")
        return [dcc.Graph(figure=fig, style=styles["graph"])]
