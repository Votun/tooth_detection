# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import base64
import io

from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.inference.inf_yolo import inference

app = Dash(__name__)

img = inference("../../", '6.png', 'models/yolo_ext.pt').plot()
fig = px.imshow(img)

app.layout = html.Div([
    html.Div(className='row', children="Dental Vision", style={'textAlign': 'center', 'fontsize': 35}),
    dcc.Upload(
            id='upload_img',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
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

    html.Div(id="graph", children="Your image will be here") #dcc.Graph(id="graph", figure=fig)
],
    style={'width': '100%','display': 'inline-block'})

@callback(
    Output('graph', component_property='children', allow_duplicate=True),
    Input('upload_img', 'contents'),
    State('upload_img', 'filename'),
    State('upload_img', 'last_modified'), config_prevent_initial_callbacks='initial_duplicate'
)
def parse_contents(contents, filename, last_modified):
    cont = contents
    cont = cont.replace("data:image/png;base64,", "")
    msg = base64.b64decode(cont)
    buf = io.BytesIO(msg)
    img = Image.open(buf).convert('RGB')
    img_rgb = np.array(img, dtype=np.uint8)
    fig = px.imshow(img_rgb)
    return html.Div([
        html.H5(filename),
        # HTML images accept base64 encoded strings in the same format
        # that is supplied by the upload
        dcc.Graph(figure=px.imshow(img_rgb)),
        html.Button(id="start_magic", children="Process image", style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },),
    ])

@callback(Output('graph', component_property='children', allow_duplicate=True),
    Input('start_magic', component_property='filename'),
    Input('upload_img', 'contents'),
    State('upload_img', 'filename'), config_prevent_initial_callbacks='initial_duplicate'
)
def process_image(contents, filename, n_clicks):
    if n_clicks:
        cont = filename
        cont = cont.replace("data:image/png;base64,", "")
        msg = base64.b64decode(cont)
        buf = io.BytesIO(msg)
        img = Image.open(buf).convert('RGB')
        predict = inference("../../", img, 'models/yolo_ext.pt')
        return html.Div([
            html.H5(n_clicks),
            # HTML images accept base64 encoded strings in the same format
            # that is supplied by the upload
            dcc.Graph(figure=px.imshow(predict.plot()))
        ])

if __name__ == '__main__':
    app.run_server(debug=True)
