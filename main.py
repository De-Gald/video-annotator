import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_player as player
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

FRAMERATE = 24

app = dash.Dash(__name__)
app.title = 'Video annotator'
server = app.server

app.scripts.config.serve_locally = True
app.config['suppress_callback_exceptions'] = True

try:
  import googleclouddebugger
  googleclouddebugger.enable(
    breakpoint_enable_canary=True
  )
except ImportError:
  pass


def load_data(path):
    # Load the dataframe containing all the processed object detections inside the video
    video_info_df = pd.read_csv(path)

    # The list of classes, and the number of classes
    classes_list = video_info_df["class_str"].value_counts().index.tolist()
    n_classes = len(classes_list)

    # Gets the smallest value needed to add to the end of the classes list to get a square matrix
    root_round = np.ceil(np.sqrt(len(classes_list)))
    total_size = root_round ** 2
    padding_value = int(total_size - n_classes)
    classes_padded = np.pad(classes_list, (0, padding_value), mode='constant')

    # The padded matrix containing all the classes inside a matrix
    classes_matrix = np.reshape(classes_padded, (int(root_round), int(root_round)))

    # Flip it for better looks
    classes_matrix = np.flip(classes_matrix, axis=0)

    data_dict = {
        "video_info_df": video_info_df,
        "n_classes": n_classes,
        "classes_matrix": classes_matrix,
        "classes_padded": classes_padded,
        "root_round": root_round
    }

    return data_dict


# Main App
app.layout = html.Div(
    children=[
        html.Div(
            className='container',
            children=[
                html.Div(
                    id='left-side-column',
                    className='seven columns',
                    style={'display': 'flex',
                           'flexDirection': 'column',
                           'flex': 1,
                           'height': 'calc(100vh - 5px)',
                           'backgroundColor': '#F9F9F9',
                           'overflow-y': 'scroll',
                           'marginLeft': '0px',
                           'justifyContent': 'flex-start',
                           'alignItems': 'center'},
                    children=[
                        html.Div(
                            id='header-section',
                            children=[
                                html.H4(
                                    'Online Video Annotator'
                                ),
                                html.P(
                                    "Выберите видео из выпадающего списка. "
                                    "После воспроизведения видео, в правой части экрана будут отображены найденные с помощью нейронной сети объекты. "
                                    "Увеличивая минимальную точность, мы задаём 'уверенность' определения объектов."
                                )
                            ]
                        ),
                        html.Div(
                            className='video-outer-container',
                            children=html.Div(
                                style={'width': '100%', 'paddingBottom': '56.25%', 'position': 'relative'},
                                children=player.DashPlayer(
                                    id='video-display',
                                    style={'position': 'absolute', 'width': '100%',
                                           'height': '100%', 'top': '0', 'left': '0', 'bottom': '0', 'right': '0'},
                                    controls=True,
                                    playing=False,
                                    volume=1,
                                    width='100%',
                                    height='100%'
                                )
                            )
                        ),
                        html.Div(
                            className='control-section',
                            children=[
                                html.Div(
                                    className='control-element',
                                    children=[
                                        html.Div(children=["Минимальная точность:"], style={'width': '40%'}),
                                        html.Div(dcc.Slider(
                                            id='slider-minimum-confidence-threshold',
                                            min=20,
                                            max=80,
                                            marks={i: f'{i}%' for i in range(20, 81, 10)},
                                            value=50,
                                            updatemode='drag'
                                        ), style={'width': '60%'})
                                    ]
                                ),

                                html.Div(
                                    className='control-element',
                                    children=[
                                        html.Div(children=["Выбор видео:"], style={'width': '40%'}),
                                        dcc.Dropdown(
                                            id="dropdown-footage-selection",
                                            options=[
                                                {'label': 'Road traffic', 'value': 'road_traffic'},
                                                {'label': 'Google home superbowl', 'value': 'google_home_superbowl'},
                                                {'label': 'Nature', 'value': 'nature'}
                                            ],
                                            value='road_traffic',
                                            clearable=False,
                                            style={'width': '60%'}
                                        )
                                    ]
                                ),

                                html.Div(
                                    className='control-element',
                                    children=[
                                        html.Div(children=["Режим показа:"], style={'width': '40%'}),
                                        dcc.Dropdown(
                                            id="dropdown-video-display-mode",
                                            options=[
                                                {'label': 'Display without Bounding Boxes', 'value': 'regular'},
                                                {'label': 'Display with Bounding Boxes', 'value': 'bounding_box'},
                                            ],
                                            value='bounding_box',
                                            searchable=False,
                                            clearable=False,
                                            style={'width': '60%'}
                                        )
                                    ]
                                ),

                            ]
                        )
                    ]
                ),
                html.Div(
                    id='right-side-column',
                    className='five columns',
                    style={
                        'height': 'calc(100vh - 5px)',
                        'overflow-y': 'scroll',
                        'marginLeft': '1%',
                        'display': 'flex',
                        'backgroundColor': '#F9F9F9',
                        'flexDirection': 'column'
                    },
                    children=[
                        dcc.Interval(
                            id="interval-visual-mode",
                            interval=700,
                            n_intervals=0
                        ),
                        html.Div(
                            children=[
                                html.P(children="Точность определения",
                                       className='plot-title'),
                                dcc.Graph(
                                    id="heatmap-confidence",
                                    style={'height': '45vh', 'width': '100%'})
                            ]
                        ),
                        html.P(children="Найденные объекты",
                               className='plot-title'),
                        html.Table(
                            id='table_1'
                        ),
                    ]
                )])
    ]
)


# Data Loading
@app.server.before_first_request
def load_all_footage():
    global data_dict, url_dict, images_dict

    # Load the dictionary containing all the variables needed for analysis
    data_dict = {
        'google_home_superbowl': load_data("data/google-home-superbowl.csv"),
        'road_traffic': load_data("data/road_traffic.csv"),
        'nature': load_data("data/nature.csv"),
    }

    url_dict = {
        'regular': {
            'google_home_superbowl': 'https://storage.googleapis.com/initial_footages/google-home-superbowl.mp4',
            'road_traffic': 'https://storage.googleapis.com/initial_footages/road_traffic.mp4',
            'nature': 'https://storage.googleapis.com/initial_footages/nature.mp4',
        },

        'bounding_box': {
            'google_home_superbowl': 'https://storage.googleapis.com/annotated_videos/google-home-superbowl.mp4',
            'road_traffic': 'https://storage.googleapis.com/annotated_videos/road_traffic.mp4',
            'nature': 'https://storage.googleapis.com/annotated_videos/nature.mp4',
        }
    }


# Footage Selection
@app.callback(Output("video-display", "url"),
              [Input('dropdown-footage-selection', 'value'),
               Input('dropdown-video-display-mode', 'value')])
def select_footage(footage, display_mode):
    # Find desired footage and update player video
    url = url_dict[display_mode][footage]
    return url


@app.callback(Output("heatmap-confidence", "figure"),
              [Input("interval-visual-mode", "n_intervals")],
              [State("video-display", "currentTime"),
               State('dropdown-footage-selection', 'value'),
               State('slider-minimum-confidence-threshold', 'value')])
def update_heatmap_confidence(n, current_time, footage, threshold):
    layout = go.Layout(
        showlegend=False,
        paper_bgcolor='rgb(249,249,249)',
        plot_bgcolor='rgb(249,249,249)',
        autosize=False,
        margin=go.layout.Margin(
            l=10,
            r=10,
            b=20,
            t=20,
            pad=4
        )
    )

    if current_time is not None:
        current_frame = round(current_time * FRAMERATE)

        if n > 0 and current_frame > 0:
            # Load variables from the data dictionary
            video_info_df = data_dict[footage]["video_info_df"]
            classes_padded = data_dict[footage]["classes_padded"]
            root_round = data_dict[footage]["root_round"]
            classes_matrix = data_dict[footage]["classes_matrix"]

            # Select the subset of the dataset that correspond to the current frame
            frame_df = video_info_df[video_info_df["frame"] == current_frame]

            # Select only the frames above the threshold
            threshold_dec = threshold / 100
            frame_df = frame_df[frame_df["score"] > threshold_dec]

            # Remove duplicate, keep the top result
            frame_no_dup = frame_df[["class_str", "score"]].drop_duplicates("class_str")
            frame_no_dup.set_index("class_str", inplace=True)

            # The list of scores
            score_list = []
            for el in classes_padded:
                if el in frame_no_dup.index.values:
                    score_list.append(frame_no_dup.loc[el][0])
                else:
                    score_list.append(0)

            # Generate the score matrix, and flip it for visual
            score_matrix = np.reshape(score_list, (-1, int(root_round)))
            score_matrix = np.flip(score_matrix, axis=0)

            # We set the color scale to white if there's nothing in the frame_no_dup
            if frame_no_dup.shape != (0, 1):
                colorscale = [[0, '#f9f9f9'], [1, '#000cff']]
            else:
                colorscale = [[0, '#f9f9f9'], [1, '#f9f9f9']]

            hover_text = [f"{score * 100:.2f}% confidence" for score in score_list]
            hover_text = np.reshape(hover_text, (-1, int(root_round)))
            hover_text = np.flip(hover_text, axis=0)

            # Add linebreak for multi-word annotation
            classes_matrix = classes_matrix.astype(dtype='|U40')

            for index, row in enumerate(classes_matrix):
                row = list(map(lambda x: '<br>'.join(x.split()), row))
                classes_matrix[index] = row

            # Set up annotation text
            annotation = []
            for y_cord in range(int(root_round)):
                for x_cord in range(int(root_round)):
                    annotation_dict = dict(
                        showarrow=False,
                        text=classes_matrix[y_cord][x_cord],
                        xref='x',
                        yref='y',
                        x=x_cord,
                        y=y_cord
                    )
                    if score_matrix[y_cord][x_cord] > 0:
                        annotation_dict['font'] = {'color': '#F9F9F9', 'size': '11'}
                    else:
                        annotation_dict['font'] = {'color': '#606060', 'size': '11'}
                    annotation.append(annotation_dict)

            # Generate heatmap figure
            figure = {
                'data': [
                    {'colorscale': colorscale,
                     'showscale': False,
                     'hoverinfo': 'text',
                     'text': hover_text,
                     'type': 'heatmap',
                     'zmin': 0,
                     'zmax': 1,
                     'xgap': 1,
                     'ygap': 1,
                     'z': score_matrix}],
                'layout':
                    {'showlegend': False,
                     'autosize': False,
                     'paper_bgcolor': 'rgb(249,249,249)',
                     'plot_bgcolor': 'rgb(249,249,249)',
                     'margin': {'l': 10, 'r': 10, 'b': 20, 't': 20, 'pad': 2},
                     'annotations': annotation,
                     'xaxis': {'showticklabels': False, 'showgrid': False, 'side': 'top', 'ticks': ''},
                     'yaxis': {'showticklabels': False, 'showgrid': False, 'side': 'left', 'ticks': ''}
                     }
            }

            return figure

    # Returns empty figure
    return go.Figure(data=[go.Pie()], layout=layout)


@app.callback(Output("table_1", "children"),
              [Input("interval-visual-mode", "n_intervals")],
              [State("video-display", "currentTime"),
               State('dropdown-footage-selection', 'value'),
               State('slider-minimum-confidence-threshold', 'value')])
def update_images(n, current_time, footage, threshold):
    if current_time is not None:
        current_frame = round(current_time * FRAMERATE)

        if n > 0 and current_frame > 0:
            video_info_df = data_dict[footage]["video_info_df"]

            # Select the subset of the dataset that correspond to the current frame
            frame_df = video_info_df[video_info_df["frame"] == current_frame]

            # Select only the frames above the threshold
            threshold_dec = threshold / 100
            frame_df = frame_df[frame_df["score"] > threshold_dec]
            children = []
            for src in list(frame_df.link):
                children.append(html.Img(
                    className='image_style',
                    src=src
                ))
            return children
    return None


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080)
