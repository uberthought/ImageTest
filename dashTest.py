#!/usr/bin/python3

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np

app = dash.Dash()

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''Dash: A web application framework for Python.'''),
    dcc.Graph(id='test-graph'),
    dcc.Slider( id='my-slider', min=0, max=5000, step=100, value=2500,),
])

@app.callback(
    dash.dependencies.Output('test-graph', 'figure'),
    [dash.dependencies.Input(component_id='my-slider', component_property='value')]
)
def update_figure(value):

    x = np.random.randn(value)
    y = np.random.randn(value)

    return go.Figure(
        data=[
            go.Scatter(
                x=x,
                y=y,
                name='test',
                mode='markers'
            )
        ],
        layout=go.Layout(
            title='Test Chart',
            showlegend=True,
            legend=go.Legend(
                x=0,
                y=1.0
            ),
            margin=go.Margin(l=40, r=0, t=40, b=30)
        )
    )

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0')
