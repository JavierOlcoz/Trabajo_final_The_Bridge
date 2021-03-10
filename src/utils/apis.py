# Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.exceptions import PreventUpdate
import glob as gb
import cv2

import numpy as np
from PIL import Image

#--------------------------------------------------------------------------------------
# Load model

model = load_model('C:\\Users\\usuario\\Desktop\\alumno_data_sciece\\data_science\\Trabajo_final_The_Bridge\\models\\VGG16_15_epochs_model.h5')


#-----------------------------------------------------------------------------

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#413BF7',
    'text': '#FFFF00',
    'button':'#ff0000'
}

#------------------------------------------------------------------------------
# inside layout 


title = html.H1(style={'color': colors['text']},
                children="The Simpson's Family Classifier")
subtitle_1 = html.H3(style={'color': colors['text']},
                children="Sideshow Bob Edition")
subtitle = html.Div(
    style={"padding-bottom": 10,
           "color": colors['text']},
    children="Click button to load the image",
)
button = html.Button(style={"backgroundColor": colors['button'], 'color': colors['text']}, children="Predict Image", id="submit-val")
space = html.Br()
sample_image = html.Img(
    style={"padding": 10, "width": "224px", "height": "224px"}, id="image"
)
model_prediction = html.H3(style={'color': colors['text']}, id="pred", children=None)
shoot = html.H3(style={'color': colors['text']}, id='who', children=None)
intermediate = html.Div(id="intermediate-operation", style={"display": 'none'})
image_bob = html.Img(
    style={"position":"fixed","top":150, "left":100,"padding": 10, "width": "300px",
    "height": "300px"}, src=app.get_asset_url('Gino.png')
)
image_bart = html.Img(
    style={"position":"fixed","top":150, "right":100,"padding": 10, "width": "300px",
    "height": "300px"}, src=app.get_asset_url('image_bart.png')
)

#--------------------------------------------------------------------------------------
# layout
app.layout = html.Div(
    style={"textAlign": "center",
           "backgroundColor": colors['background']},
    children=[
        space,
        title,
        subtitle_1,
        subtitle,
        button,
        space,
        sample_image,
        model_prediction,
        shoot,
        intermediate,
        space,
        space,
        space,
        space,
        image_bob,
        image_bart
    ])


#----------------------------------------------------
# Callbacks

import random

NUM_IMAGES = 31

@app.callback(
    dash.dependencies.Output("intermediate-operation", "children"),
    [dash.dependencies.Input("submit-val", "n_clicks")],
)
def update_random_image(n_clicks):
    if n_clicks is None:
        raise PreventUpdate
    else:
        return random.choice(range(NUM_IMAGES))

@app.callback(
    dash.dependencies.Output("image", "src"),
    [dash.dependencies.Input("intermediate-operation", "children")],
)
def get_image(argu):
    z = str(argu) + '.jpg'
    
    return app.get_asset_url(z)


@app.callback(
    dash.dependencies.Output("pred", "children"),
    [dash.dependencies.Input("intermediate-operation", "children")],
)

def update_prediction(path):
    s = 224

    X_test = []
    file = 'C:\\Users\\usuario\\Desktop\\alumno_data_sciece\\data_science\\Trabajo_final_The_Bridge\\src\\utils\\assets\\' + str(path) + '.jpg'

         
    image = cv2.imread(file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_array = cv2.resize(image , (s,s))
    X_test.append(list(image_array))

    X_test = np.array(X_test, dtype = 'float32')
    X_test = X_test/255.0
    predictions = model.predict(X_test)
    predictions_1 = np.argmax(predictions, axis=-1)
    if predictions_1 == 0:
        predictions_1 = 'Abuelo Simpson'
    elif predictions_1 == 1:
        predictions_1 = 'Bart Simpson'
    elif predictions_1 == 2:
        predictions_1 = 'Homer Simpson'
    elif predictions_1 == 3:
        predictions_1 = 'Lisa Simpson'
    elif predictions_1 == 4:
        predictions_1 = 'Maggie Simpson'
    elif predictions_1 == 5:
        predictions_1 = 'Marge Simpson'
    
    return "Prediction: " + str(predictions_1)

@app.callback(
    dash.dependencies.Output("who", "children"),
    [dash.dependencies.Input("pred", "children")],
)

def shoot_1(arg):
    if arg == "Prediction: Bart Simpson":
        x = 'SHOOT!!!'
    else:
        x = 'DONT SHOOT!!'
    return x

if __name__ == "__main__":
    app.run_server(debug=True)