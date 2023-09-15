import dash
from dash import dcc
from dash import html
import pandas as pd
import plotly.express as px
import json
from glob import glob
from multiprocessing import Pool, RLock
import pandas as pd
import multiprocessing as mp
import seaborn as sn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import dash_bootstrap_components as dbc
from dash import html
import statistics
from statistics import mean
import math
import os
import dash_daq as daq
import json
import numpy as np
import pandas as pd

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server
with open('manual.txt', 'r', encoding='utf-8') as f:
    manual = [line.rstrip() for line in f]
with open('datasets.json', 'r', encoding='utf-8') as f:
    datasets = json.load(f)

lang_file = pd.read_csv('all_languages.csv', delimiter=';')
genealogy = pd.read_csv('genealogy.csv', delimiter=',')
languages_for_map = [{'label': str(i), 'value': str(i)} for i in genealogy['language'].tolist()]
df_data = pd.read_csv('data.csv', delimiter=',')

def euc_dist(pt1, pt2):
    return math.sqrt((pt2[0]-pt1[0])*(pt2[0]-pt1[0])+(pt2[1]-pt1[1])*(pt2[1]-pt1[1]))
def _c(ca,i,j,P,Q):
    if ca[i,j] > -1:
        return ca[i,j]
    elif i == 0 and j == 0:
        ca[i,j] = euc_dist(P[0],Q[0])
    elif i > 0 and j == 0:
        ca[i,j] = max(_c(ca,i-1,0,P,Q),euc_dist(P[i],Q[0]))
    elif i == 0 and j > 0:
        ca[i,j] = max(_c(ca,0,j-1,P,Q),euc_dist(P[0],Q[j]))
    elif i > 0 and j > 0:
        ca[i,j] = max(min(_c(ca,i-1,j,P,Q),_c(ca,i-1,j-1,P,Q),_c(ca,i,j-1,P,Q)),euc_dist(P[i],Q[j]))
    else:
        ca[i,j] = float("inf")
    return ca[i,j]
 
 # This is the method called for us
def frechet_distance(P,Q):
    ca = np.ones((len(P),len(Q)))
    ca = np.multiply(ca,-1)
    return _c(ca, len(P)-1, len(Q)-1, P, Q) # ca is the matrix of a*b (3*4), 2, 3
   
@app.callback(
    Output('modal', 'is_open'),
    [Input('open', 'n_clicks'), Input('close', 'n_clicks')],
    [State('modal', 'is_open')],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

@app.callback(
    [   
        Output(component_id='metric_selection', component_property='options'),
        Output(component_id='metric_selection', component_property='value'),
    ],
    Input(component_id='model_selection', component_property='value'), 
)
def update_output(model_name):
    metrics = df_data[df_data['hf_model_name'] == model_name]['metric_names'].unique().tolist()
    return [{'label': str(i), 'value': str(i)} for i in metrics], metrics[0]

@app.callback(
    Output(component_id='map', component_property='figure'),
    Input(component_id='language_selection_map', component_property='value'),
)
def update_output(language):
    result = []
    cards_info = {}
    if isinstance(language, list):
        for l in language:
            info = genealogy.loc[genealogy['language'].isin([l])]
            result.append([l, info['family'].iloc[0], info['subfamily'].iloc[0], info['genus'].iloc[0], info['coordinates'].iloc[0]])
            cards_info[l] = {}
            cards_info[l]['Family'] = info['family'].iloc[0]
            if info['subfamily'].iloc[0]:
                cards_info[l]['Subfamily'] = info['subfamily'].iloc[0]
            if info['genus'].iloc[0]:
                cards_info[l]['Genus'] = info['genus'].iloc[0]
    else:
        info = genealogy.loc[genealogy['language'].isin([language])]
        result = [[language, info['family'].iloc[0], info['subfamily'].iloc[0], info['genus'].iloc[0], info['coordinates'].iloc[0]]]
        cards_info[language] = {}
        cards_info[language]['Family'] = info['family'].iloc[0]
        if info['subfamily'].iloc[0]:
            cards_info[language]['Subfamily'] = info['subfamily'].iloc[0]
        if info['genus'].iloc[0]:
            cards_info[language]['Genus'] = info['genus'].iloc[0]
    df_map = pd.DataFrame(columns = ['Language', 'Family', 'Subfamily', 'Genus', 'Latitude', 'Longitude'])
    count = 0
    for res in result:
        count += 1

        temp = {'Language': res[0], 'Family': res[1], 'Subfamily': res[2], 'Genus': res[3], 'Latitude': res[4][2:-2].split(', ')[0], 'Longitude': res[4][2:-2].split(', ')[1]}
        df_temp = pd.DataFrame(temp, index=[count])
        frames = [df_map, df_temp]
        df_map = pd.concat(frames)

    df_map['Latitude'] = df_map['Latitude'].apply(pd.to_numeric)
    df_map['Longitude'] = df_map['Longitude'].apply(pd.to_numeric)
    df_map = df_map.fillna('no')

    fig = px.scatter_mapbox(df_map, lat='Latitude', lon='Longitude', hover_name='Language', hover_data=['Family', 'Subfamily', 'Genus', 'Latitude', 'Longitude'],
        color_discrete_sequence=["fuchsia"], zoom=1, height=300, center = {"lat": 39, "lon": 34})
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    cards = []
    for language in cards_info.keys():
        body_text = []
        body_text.append(html.H4(f'{language}'))
        for feature in cards_info[language].keys():
            body_text.append(f'{feature}: {cards_info[language][feature]}')
        card = dbc.Card(
            [
                dbc.CardBody(
                    dbc.Row([
                            a for a in body_text
                    ])
                ),
            ],
            body=True,
            )
        cards.append(card)
    return fig

@app.callback(
    [
        Output(component_id='graph1', component_property='figure'),
        Output(component_id='graph2', component_property='figure'),
    ],
    [
    Input(component_id='model_selection', component_property='value'),
    Input(component_id='metric_selection', component_property='value'),
    ]
)
def update_output(model_name, metric_name):
    df = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]

    df1 = pd.DataFrame(columns = ['task_category', 'Mean value'])
    df1['task_category'] = list(df.groupby(['task_category']).groups.keys())
    df1['Mean value'] = df.groupby(['task_category']).agg(['mean'])['metric_scores']['mean'].tolist()
    
    fig1 = px.bar(df1, x='Mean value', y='task_category', orientation="h", height=750)
    fig1.update_layout(showlegend=False)
    fig1.update_layout(yaxis={'categoryorder':'total ascending'})
   
    df2 = pd.DataFrame(columns = ['Family', 'Mean value'])
    df2['Family'] = list(df_data.groupby(['family']).groups.keys())
    df2['Mean value'] = df_data.groupby(['family']).agg(['mean'])['metric_scores']['mean'].tolist()
    df2['Size'] = df_data.groupby(['family'])['full_language'].nunique().tolist()
    try:
        fig2 = px.scatter(df2,
            x='Family',
            y='Mean value', 
            size='Size'
        )
        fig2.update_layout(hovermode='x unified')
    except:
        fig2 = go.Figure()
    fig2.update_layout({
    'paper_bgcolor': 'rgba(0,0,0,0)'
    })
    return fig1, fig2

@app.callback(
    Output(component_id='note_graph2', component_property='children'),
    [
    Input(component_id='model_selection', component_property='value'),
    Input(component_id='metric_selection', component_property='value'),
    ]
)
def update_output(model_name, metric_name):
    res = str()
    df = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]
    if '[Basque]*' in list(df.groupby(['family']).groups.keys()):
       res = '*Basque is an isolate language, it is not included in any of the language groups'
    return res

@app.callback(
    Output(component_id='card_info', component_property='children'),
    [
    Input(component_id='model_selection', component_property='value'),
    Input(component_id='metric_selection', component_property='value'),
    ]
)
def update_output(model_name, metric_name):
    df = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]
    df = df[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)].groupby(['family', 'full_language'])
    df = df.agg({"task_category": "nunique"}).reset_index()
    families = list(df.groupby(['family']).groups.keys())
    numbers = df.groupby(['family']).sum()['task_category'].tolist()
    size = {}
    for i in range(len(families)):
        size[families[i]] = numbers[i]
    sorted_size = dict(sorted(size.items(), key=lambda item: item[1], reverse=True))
    content = [html.H4('Number of files', style={"font-weight": "bold"})]
    for key in sorted_size.keys():
        description = f"{key}: {sorted_size[key]}"
        content.append(html.P(f'{description}'))
    card_info = dbc.Card(
        [
            dbc.CardBody(content),
        ],
        body=True,
        className="border-0 bg-transparent"
    )
    return card_info

@app.callback(
    Output(component_id='graph3', component_property='figure'),
    [
    Input(component_id='model_selection', component_property='value'),
    Input(component_id='metric_selection', component_property='value'),
    ]
)
def update_output(model_name, metric_name):
    df = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]
    info = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)].groupby(['layer']).describe()['metric_scores']['mean'].to_dict()
    info = dict(sorted(info.items(), key=lambda item: int(item[0]), reverse=False))
    df1 = pd.DataFrame(columns=['Layer', 'Average value'])
    df1['Layer'] = info.keys(); df1['Average value'] = info.values()    
    
    tr1 = px.bar(df1, x='Layer', y='Average value', text_auto=True, title=model_name)
    tr1['layout'].update(yaxis_range=[0,1])
    tr1.update_traces(marker_color='rgb(31, 58, 193)')

    tr2 = px.box(df, x= 'layer', y='metric_scores')
    tr2.update_traces(marker_color='rgb(133, 150, 232)')
    fig = go.Figure(data=tr1.data + tr2.data)
    if len(df['layer'].tolist()) < 30:
        fig.update_layout(
            xaxis = dict(tickmode = 'linear', dtick = 1),
        )
    return fig

@app.callback(
    Output(component_id='graph4', component_property='figure'),
    [
    Input(component_id='model_selection', component_property='value'),
    Input(component_id='metric_selection', component_property='value'),
    ]
)
def update_output(model_name, metric_name):
    df = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]
    info = df.groupby(['full_language']).describe()['metric_scores']['mean'].to_dict()
    info = dict(sorted(info.items(), key=lambda item: item[1], reverse=True))
    df1 = pd.DataFrame(columns=['full_language', 'Average value'])
    df1['full_language'] = info.keys(); df1['Average value'] = info.values()

    tr1 = px.bar(df1, x='full_language', y='Average value', text_auto=True, title=model_name)
    tr1['layout'].update(yaxis_range=[0,1])
    tr1.update_traces(marker_color='rgb(126, 63, 175)')

    tr2 = px.box(df1, x= 'full_language', y='Average value')
    tr2.update_traces(marker_color='rgb(148, 109, 255)')
    fig = go.Figure(data=tr1.data + tr2.data)
    return fig

@app.callback(
    [   
        Output(component_id='graph5_dropdown', component_property='options'),
        Output(component_id='graph5_dropdown', component_property='value'),
    ],
    [
        Input(component_id='model_selection', component_property='value'),
        Input(component_id='metric_selection', component_property='value'),
    ]
)
def update_output(model_name, metric_name):
    list_of_families = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]['family'].unique().tolist()
    return [{'label': str(i), 'value': str(i)} for i in list_of_families], list_of_families[0]

@app.callback(
    Output(component_id='graph5', component_property='figure'),
    [
        Input(component_id='model_selection', component_property='value'),
        Input(component_id='metric_selection', component_property='value'),
        Input(component_id='graph5_dropdown', component_property='value'),
    ]
)
def update_output(model_name, metric_name,families):
    df = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]
    df_graph5 = pd.DataFrame(columns = ['Family', 'X', 'Y'])
    if isinstance(families, list):
        df = df[df['family'].isin(families)]
    else:
        df = df[df['family'].isin([families])]
    df_temp = df.groupby(['family', 'layer']).describe()['metric_scores']['mean']
    df_temp = dict(df_temp.items())
    family = []
    x = []
    y = []
    for key in df_temp.keys():
        family.append(key[0]); x.append(key[1]); y.append(df_temp[key])
    df_graph5['Family'] = family; df_graph5['X'] = x; df_graph5['Y'] = y
    try:
        fig5 = px.line(df_graph5, x = 'X', y = 'Y', color='Family', labels={'X':'Layer number', 'Y': 'Average value'})
    except:
        fig5 = go.Figure()
    fig5.update_layout({
    'paper_bgcolor': 'rgba(0,0,0,0)'
    })
    return fig5

@app.callback(
    [   Output(component_id='graph6_category', component_property='options'),
        Output(component_id='graph6_category', component_property='value'),
    ],
    [
        Input(component_id='model_selection', component_property='value'),
        Input(component_id='metric_selection', component_property='value'),
    ]
)
def update_output(model_name, metric_name):
    all_cats = sorted(df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]['task_category'].unique().tolist())
    return [{'label': str(i), 'value': str(i)} for i in all_cats], all_cats[0]

@app.callback(
    [   
        Output(component_id='graph6_languages', component_property='options'),
        Output(component_id='graph6_languages', component_property='value'),
    ],
    [
        Input(component_id='model_selection', component_property='value'),
        Input(component_id='metric_selection', component_property='value'),
        Input(component_id='graph6_category', component_property='value'),
    ]
)
def update_output(model_name, metric_name, category):
    if category:
        list_of_languages = sorted(df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name) & (df_data['task_category'] == category)]['full_language'].unique().tolist())
    else:
        list_of_languages = sorted(df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]['full_language'].unique().tolist())
    return [{'label': str(i), 'value': str(i)} for i in list_of_languages], list_of_languages[0]

@app.callback(
    Output(component_id='graph6', component_property='figure'),
    [   
        Input(component_id='model_selection', component_property='value'),
        Input(component_id='metric_selection', component_property='value'),
        Input(component_id='graph6_category', component_property='value'),
        Input(component_id='graph6_languages', component_property='value'),
    ],
)
def update_output(model_name, metric_name, category, languages):
    if category:
        df = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name) & (df_data['task_category'] == category)]
        if isinstance(languages, list):
            df_graph6 = df[df['full_language'].isin(languages)]
        elif isinstance(languages, str):
            df_graph6 = df[df['full_language'].isin([languages])]
        try:
            fig6 = px.line(df_graph6, x = 'layer', y = 'metric_scores', color='full_language',labels={'layer':'Layer number', 'metric_scores': 'Value'})
        except:
            fig6 = go.Figure()
    else:
        fig6 = go.Figure()
    fig6.update_layout({'paper_bgcolor': 'rgba(0,0,0,0)'
    })
    return fig6

@app.callback(
    [   
        Output(component_id='graph7_language', component_property='options'),
        Output(component_id='graph7_language', component_property='value'),
    ],
    [   
        Input(component_id='model_selection', component_property='value'),
        Input(component_id='metric_selection', component_property='value'),
    ],
)
def update_output(model_name, metric_name):
    list_of_languages = sorted(df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]['full_language'].unique().tolist())
    return [{'label': str(i), 'value': str(i)} for i in list_of_languages], list_of_languages[0]

@app.callback(
    [   
        Output(component_id='graph7_categories', component_property='options'),
        Output(component_id='graph7_categories', component_property='value'),
    ],
    [   
        Input(component_id='model_selection', component_property='value'),
        Input(component_id='metric_selection', component_property='value'),
        Input(component_id='graph7_language', component_property='value'),
    ],
)
def update_output(model_name, metric_name, language):
    if language:
        list_of_categories = sorted(df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name) & (df_data['full_language'] == language)]['task_category'].unique().tolist())
    else:
        list_of_categories = sorted(df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]['task_category'].unique().tolist())    
    return [{'label': str(i), 'value': str(i)} for i in list_of_categories], list_of_categories[0]

@app.callback(
    Output(component_id='graph7', component_property='figure'),
    [   
        Input(component_id='model_selection', component_property='value'),
        Input(component_id='metric_selection', component_property='value'),
        Input(component_id='graph7_language', component_property='value'),
        Input(component_id='graph7_categories', component_property='value'),
    ],
)
def update_output(model_name, metric_name, language, categories):
    df_graph7 = pd.DataFrame()
    if language:    
        df_graph7 = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name) & (df_data['full_language'] == language)]
        if isinstance(categories, list):
            df_graph7 = df_graph7[df_graph7['task_category'].isin(categories)]
        else:
            df_graph7 = df_graph7[df_graph7['task_category'].isin([categories])]
    if df_graph7.empty:
        fig7 = go.Figure()
    else:
        fig7 = px.line(df_graph7, x = 'layer', y = 'metric_scores', color='task_category', labels={'layer':'Layer number', 'metric_scores': 'Value'})
    fig7.update_layout({'paper_bgcolor': 'rgba(0,0,0,0)'})
    return fig7

@app.callback(
    Output(component_id='graph7_datasets', component_property='children'),
    [   
        Input(component_id='model_selection', component_property='value'),
        Input(component_id='graph7_language', component_property='value'),
        Input(component_id='graph7_categories', component_property='value'),
    ],
)
def update_output(model_name, language, categories):
    cards = []
    if language:
        if isinstance(categories, list):
            for category in categories:
                body_text = []  
                body_text.append(html.H4(f'{category}', style={'font-weight': 'bold'})),
                training = [html.B("Training:")]
                dataset_size = int()
                for key in datasets[model_name][language][category]['training'].keys():
                    training.append(html.P(f"{key} — {datasets[model_name][language][category]['training'][key]}"))
                    dataset_size += int(datasets[model_name][language][category]['training'][key])
                validation = [html.B("Validation:")]
                for key in datasets[model_name][language][category]['validation'].keys():
                    validation.append(html.P(f"{key} — {datasets[model_name][language][category]['validation'][key]}"))
                    dataset_size += int(datasets[model_name][language][category]['validation'][key])
                test = [html.B("Test:")]
                for key in datasets[model_name][language][category]['test'].keys():
                    test.append(html.P(f"{key} — {datasets[model_name][language][category]['test'][key]}"))
                    dataset_size += int(datasets[model_name][language][category]['test'][key])
                body_text.append(dbc.Col([
                                        a for a in training
                                        ])
                )
                body_text.append(dbc.Col([
                                        a for a in validation
                                        ])
                )
                body_text.append(dbc.Col([
                                        a for a in test
                                        ])
                )

                card = dbc.Card(
                    [
                        dbc.CardBody([
                            dbc.Row([
                                    a for a in body_text
                            ]),
                            html.Div([
                                html.B('In total:'),
                                html.P(f'{dataset_size}')
                            ]),
                        ]),
                    ],
                    className='bg-transparent',
                    style={'color': 'black'},
                    body=True,
                )
                cards.append(card)

        else:
            category = categories
            body_text = []  
            body_text.append(html.H4(f'{category}', style={'font-weight': 'bold'}))
            training = [html.B("Training:")]
            dataset_size = int()
            for key in datasets[model_name][language][category]['training'].keys():
                training.append(html.P(f"{key} — {datasets[model_name][language][category]['training'][key]}"))
                dataset_size += int(datasets[model_name][language][category]['training'][key])
            validation = [html.B("Validation:")]
            for key in datasets[model_name][language][category]['validation'].keys():
                validation.append(html.P(f"{key} — {datasets[model_name][language][category]['validation'][key]}"))
                dataset_size += int(datasets[model_name][language][category]['validation'][key])
            test = [html.B("Test:")]
            for key in datasets[model_name][language][category]['test'].keys():
                test.append(html.P(f"{key} — {datasets[model_name][language][category]['test'][key]}"))
                dataset_size += int(datasets[model_name][language][category]['test'][key])
            
            body_text.append(dbc.Col([
                                    a for a in training
                                    ])
            )
            body_text.append(dbc.Col([
                                    a for a in validation
                                    ])
            )
            body_text.append(dbc.Col([
                                    a for a in test
                                    ])
            )

            card = dbc.Card(
                [   
                    dbc.CardBody([
                        dbc.Row([
                                a for a in body_text
                        ]),
                        html.Div([
                            html.B('In total:'),
                            html.P(f'{dataset_size}')
                        ]),
                    ]),
                ],
                className='bg-transparent',
                style={'color': 'black'},
                body=True,
            )
            cards.append(card)
    else:
        card = dbc.Card(
            className='bg-transparent',
            style={'color': 'black'},
            body=True,
        )
        cards.append(card)
    children = [
                dbc.Row([
                    a for a in cards
                ])
    ]
    return children

@app.callback(
    [   
        Output(component_id='family_selection', component_property='options'),
        Output(component_id='family_selection', component_property='value'),
    ],
    [   
        Input(component_id='model_selection', component_property='value'),
        Input(component_id='metric_selection', component_property='value'),
    ],
)
def update_output(model_name, metric_name):
    list_of_families = sorted(df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]['family'].unique().tolist())
    if 'Basque' in list_of_families:
        list_of_families.remove('Basque')
    return [{'label': str(i), 'value': str(i)} for i in sorted(list_of_families)], sorted(list_of_families)[0]

@app.callback(
    Output(component_id='card1', component_property='children'),
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='metric_selection', component_property='value'),
        Input(component_id='family_selection', component_property='value')
    ]
)
def update_output(model_name, metric_name, family):
    if not family:
        list_of_families = sorted(df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]['family'].unique().tolist())
        if 'Basque' in list_of_families:
            list_of_families.remove('Basque')
        family = sorted(list_of_families)[0]
    languages = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name) & (df_data['family'] == family)]['full_language'].unique().tolist()
    card1 = dbc.Card(
        [
            dbc.CardBody(
            [
                html.H2(f'{len(languages)}'),
                html.P('Number of languages in this language family', className='card-text'),
            ]
            ),
        ],
        body=True,
        color='#2D2D2D',
        inverse=True,
    )
    return card1

@app.callback(
    Output('card2', 'children'),
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='metric_selection', component_property='value'),
        Input(component_id='family_selection', component_property='value')
    ]
)
def update_output(model_name, metric_name, family):
    if not family:
        list_of_families = sorted(df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]['family'].unique().tolist())
        if 'Basque' in list_of_families:
            list_of_families.remove('Basque')
        family = sorted(list_of_families)[0]

    mean = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name) & (df_data['family'] == family)].groupby(['task_category']).describe()['metric_scores']['mean'].to_dict()
    min_count = min(mean.items(), key=lambda x: x[1])
    card2 = dbc.Card(
        [                       
            dbc.CardBody(
            [
                html.H2(f'{str(min_count[0])}'),
                html.P('The most poorly recognized category by the model', className='card-text'),      
            ]
            ),
        ],
        body=True,
        color='#2D2D2D',
        inverse=True,
    )
    return card2

@app.callback(
    Output('card3', 'children'),
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='metric_selection', component_property='value'),
        Input(component_id='family_selection', component_property='value')
    ]
)
def update_output(model_name, metric_name, family):
    if not family:
        list_of_families = sorted(df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]['family'].unique().tolist())
        if 'Basque' in list_of_families:
            list_of_families.remove('Basque')
        family = sorted(list_of_families)[0]

    mean = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name) & (df_data['family'] == family)].groupby(['task_category']).describe()['metric_scores']['mean'].to_dict()
    max_count = max(mean.items(), key=lambda x: x[1])
    card3 = dbc.Card(
        [    
            dbc.CardBody(
            [
                html.H2(f'{str(max_count[0])}'),
                html.P('The most well recognized category by the model', className='card-text'),
            ]
            ),
        ],
        body=True,
        color='#2D2D2D',
        inverse=True,
    )
    return card3

@app.callback(
    Output('treemap1', 'figure'),
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='metric_selection', component_property='value'),
        Input(component_id='family_selection', component_property='value')
    ]
)
def update_output(model_name, metric_name, family):
    if not family:
        list_of_families = sorted(df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]['family'].unique().tolist())
        if 'Basque' in list_of_families:
            list_of_families.remove('Basque')
        family = sorted(list_of_families)[0]

    labels = []; labels.append(family)
    parents = []; parents.append('')
    values = []; values.append(0)

    languages = sorted(df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name) & (df_data['family'] == family)]['full_language'].unique().tolist())
    for language in languages:
        count = len(df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name) & (df_data['family'] == family) & (df_data['full_language'] == language)])
        labels.append(language)
        parents.append(family)
        values.append(count)

    values[0] = sum(values)
    treemap1 = go.Figure(go.Treemap(
        labels = labels,
        parents= parents,
        values= values,
        root = None)
    )
    treemap1.update_layout(
        font_size=20,
        margin = dict(t=20, l=15, r=20, b=20),
    )
    treemap1.update_layout({'paper_bgcolor': 'rgba(0,0,0,0)'})
    return treemap1

@app.callback(
    Output('boxplot', 'figure'),
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='metric_selection', component_property='value'),
        Input(component_id='family_selection', component_property='value'),
        Input(component_id='tick', component_property='on')
    ]
)
def update_output(model_name, metric_name, family, tick):
    if not family:
        list_of_families = sorted(df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]['family'].unique().tolist())
        if 'Basque' in list_of_families:
            list_of_families.remove('Basque')
        family = sorted(list_of_families)[0]

    df = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name) & (df_data['family'] == family)]
    boxplot = pd.DataFrame(columns = ['full_language', 'task_category', 'average_value'])
    dict_temp = df.groupby(['task_category', 'full_language']).describe()['metric_scores']['mean'].to_dict()
    dict_temp = {k:round(v,3) for k, v in dict_temp.items()}
    categories = []
    languages = []
    y = []
    for key in dict_temp.keys():
        categories.append(key[0]); languages.append(key[1]); y.append(dict_temp[key])
    boxplot['task_category'] = categories; boxplot['full_language'] = languages; boxplot['average_value'] = y
    
    x = "{}".format(tick)
    if x == 'True':
        tr1 = px.box(boxplot, x= 'task_category', y='average_value')
        tr2 = px.scatter(boxplot, x='task_category', y='average_value', color='full_language')
        fig = go.Figure(data=tr1.data + tr2.data)
    else:
        tr1 = px.box(boxplot, x= 'task_category', y='average_value')
        fig = go.Figure(tr1.data)
    fig.update_layout({'paper_bgcolor': 'rgba(0,0,0,0)'})
    return fig

@app.callback(
    [   
        Output(component_id='quantity_of_languages', component_property='options'),
        Output(component_id='quantity_of_languages', component_property='value'),
        Output(component_id='quantity_of_languages', component_property='style'),
    ],
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='metric_selection', component_property='value'),
        Input(component_id='family_selection', component_property='value'),
    ],
)
def update_output(model_name, metric_name, family):
    if not family:
        list_of_families = sorted(df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]['family'].unique().tolist())
        if 'Basque' in list_of_families:
            list_of_families.remove('Basque')
        family = sorted(list_of_families)[0]

    family_structure = sorted(df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name) & (df_data['family'] == family)]['full_language'].unique().tolist())
    numbers = []
    df = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name) & (df_data['family'] == family)]
    flag = 0
    for category in sorted(df['task_category'].unique().tolist()):
        check = df[df['task_category'] == category]
        numbers.append(len(check['full_language'].unique().tolist()))
        if len(check['full_language'].unique().tolist()) > 3:
            flag = 1
    if len(family_structure) > 1:
        quantity = [i for i in range(1, max(numbers)+1)]
    else:
        quantity = [1]
    if flag == 0:
        style={'display': 'none'}
    else:
        style={'visibility':'visible'}
    return [{'label': int(i), 'value': int(i)} for i in quantity], quantity[0], style

@app.callback(
    Output('graphs_for_family', 'children'),
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='metric_selection', component_property='value'),
        Input(component_id='family_selection', component_property='value'),
        Input(component_id='quantity_of_languages', component_property='value')
    ],
)
def update_output(model_name, metric_name, family, quantity):
    if not family:
        list_of_families = sorted(df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]['family'].unique().tolist())
        if 'Basque' in list_of_families:
            list_of_families.remove('Basque')
        family = sorted(list_of_families)[0]

    df_similar_graph1 = pd.DataFrame(columns = ['full_language', 'task_category', 'layer', 'metric_scores'])
    df_dissimilar_graph2 = pd.DataFrame(columns = ['full_language', 'task_category', 'layer', 'metric_scores'])
    df_incomparable_graph3 = pd.DataFrame(columns = ['full_language', 'task_category', 'layer', 'metric_scores'])
    distances = {}

    df = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name) & (df_data['family'] == family)]
    for category in df['task_category'].unique().tolist():
        if len(df[df['task_category'] == category]['full_language'].unique().tolist()) > 4:
            curves = {}
            curves[category] = {}
            for index, raw in df[df['task_category'] == category].iterrows():
                if raw['full_language'] not in curves[category].keys():
                    curves[category][raw['full_language']] = []
                point = (raw['layer'], raw['metric_scores'])
                curves[category][raw['full_language']].append(point)

            distances[category] = {}
            pattern_line = []
            nam_line = list(curves[category].keys())
            for b in range(len(df[df['task_category'] == category]['layer'].unique().tolist())):
                for_pattern = []
                for language in curves[category].keys():
                    for_pattern.append(curves[category][language][b][1])
                for_median = sorted(for_pattern)
                median = statistics.median(for_median)
                pattern_line.append((b, round(median, 3)))
            for name in nam_line:
                distances[category][name] = frechet_distance(curves[category][name], pattern_line)
            distances1 = dict(sorted(distances[category].items(), key=lambda x: -x[1], reverse=True))
            distances2 = dict(sorted(distances[category].items(), key=lambda x: -x[1]))

            distances1_names = list(distances1.keys())
            distances2_names = list(distances2.keys())            
            if quantity:
                if quantity > 1:
                    for b in range(quantity):
                        if b >= len(distances1_names):
                            break
                        df_temp = df[df['full_language'] == distances1_names[b]]
                        df_temp = df_temp.loc[df_temp['task_category'].isin([category])]
                        frames = [df_similar_graph1, df_temp]
                        df_similar_graph1 = pd.concat(frames)

                        df_temp = df[df['full_language'] == distances2_names[b]]
                        df_temp = df_temp.loc[df_temp['task_category'].isin([category])]
                        frames = [df_dissimilar_graph2, df_temp]
                        df_dissimilar_graph2 = pd.concat(frames)
                else:
                    df_temp = df[df['full_language'] == distances1_names[0]]
                    df_temp = df_temp.loc[df_temp['task_category'].isin([category])]
                    frames = [df_similar_graph1, df_temp]
                    df_similar_graph1 = pd.concat(frames)

                    df_temp = df[df['full_language'] == distances2_names[0]]
                    df_temp = df_temp.loc[df_temp['task_category'].isin([category])]
                    frames = [df_dissimilar_graph2, df_temp]
                    df_dissimilar_graph2 = pd.concat(frames)

        else:
            df_temp = df[df['task_category'] == category]
            frames = [df_incomparable_graph3, df_temp]
            df_incomparable_graph3 = pd.concat(frames)

    graphs_for_div = []
    similar_graphs_for_col = []
    dissimilar_graphs_for_col = []
    incomparable_graphs_for_col = []
    if df_similar_graph1.empty == False:
        for category in df_similar_graph1['task_category'].unique().tolist():
            df_temp = df_similar_graph1[df_similar_graph1['task_category'].isin([category])]
            fig1 = px.line(df_temp, x='layer', y='metric_scores', color='full_language', 
                        labels={'layer':'Layer number', 'metric_scores': 'Value', 'full_language': 'Language'}, title=f'<b>{category}</b>')
            fig1['layout'].update(height=350)
            fig1['layout'].update(yaxis_range=[0,1])
            fig1.update_layout({
                'paper_bgcolor': 'rgba(0,0,0,0)'
            })
            row = dcc.Graph(figure=fig1)
            if len(similar_graphs_for_col) == 0:
                similar_graphs_for_col.append(html.H5('The most similar trends by category:',  style={'font-weight': 'bold', 'text-align': 'center'}))
            similar_graphs_for_col.append(row)
    if df_dissimilar_graph2.empty == False:
        for category in df_dissimilar_graph2['task_category'].unique().tolist():
            df_temp = df_dissimilar_graph2[df_dissimilar_graph2['task_category'].isin([category])]
            fig2 = px.line(df_temp, x='layer', y='metric_scores', color='full_language', 
                        labels={'layer':'Layer number', 'metric_scores': 'Value', 'full_language': 'Language'}, title=f'<b>{category}</b>')
            fig2['layout'].update(height=350)
            fig2['layout'].update(yaxis_range=[0,1])
            fig2.update_layout({
                'paper_bgcolor': 'rgba(0,0,0,0)'
            })
            row = dcc.Graph(figure=fig2)
            if len(dissimilar_graphs_for_col) == 0:
                dissimilar_graphs_for_col.append(html.H5('The most dissimilar trends by category:', style={'font-weight': 'bold', 'text-align': 'center'}))
            dissimilar_graphs_for_col.append(row)
            
    if df_incomparable_graph3.empty == False:
        for category in df_incomparable_graph3['task_category'].unique().tolist():
            df_temp = df_incomparable_graph3[df_incomparable_graph3['task_category'].isin([category])]
            fig3 = px.line(df_temp, x='layer', y='metric_scores', color='full_language', 
                        labels={'layer':'Layer number', 'metric_scores': 'Value', 'full_language': 'Language'}, title=f'<b>{category}</b>')
            fig3['layout'].update(height=350)
            fig3['layout'].update(yaxis_range=[0,1])
            fig3.update_layout({
                'paper_bgcolor': 'rgba(0,0,0,0)'
            })
            row = dcc.Graph(figure=fig3)
            if len(incomparable_graphs_for_col) == 0:
                incomparable_graphs_for_col.append(html.H5('Comparison is impossible, the number of languages is too small',  style={'font-weight': 'bold', 'text-align': 'center'}))
            incomparable_graphs_for_col.append(row)

    if len(similar_graphs_for_col) != 0:
        graphs_for_div.append(dbc.Col([a for a in similar_graphs_for_col], width=4))
    if len(dissimilar_graphs_for_col) != 0:
        graphs_for_div.append(dbc.Col([a for a in dissimilar_graphs_for_col], width=4))
    if len(incomparable_graphs_for_col) != 0:
        graphs_for_div.append(dbc.Col([a for a in incomparable_graphs_for_col], width=4))

    children = [
                dbc.Row([
                    a for a in graphs_for_div
                ])
                ]
    graphs = html.Div(children=children)
    return graphs

@app.callback(
    Output(component_id='heatmap1', component_property='figure'),
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='metric_selection', component_property='value'),
    ],
)
def update_output(model_name, metric_name):
    df = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]
    heatmap_df = pd.DataFrame(columns = ['full_language', 'task_category', 'average_value'])
    dict_temp = df.groupby(['task_category', 'full_language']).describe()['metric_scores']['mean'].to_dict()
    dict_temp = {k:round(v,3) for k, v in dict_temp.items()}
    categories = []
    languages = []
    y = []
    for key in dict_temp.keys():
        categories.append(key[0]); languages.append(key[1]); y.append(dict_temp[key])
    heatmap_df['task_category'] = categories; heatmap_df['full_language'] = languages; heatmap_df['average_value'] = y

    heatmap_df = heatmap_df.sort_values('task_category', ascending=False)
    heatmap = go.Figure(
        layout=go.Layout(
            height=1200,
        )
    )
    heatmap.add_trace(
        go.Heatmap(
        name="Number",
        y = heatmap_df['task_category'].tolist(),
        x = heatmap_df['full_language'].tolist(),
        z = np.array(heatmap_df['average_value'].tolist()),
        xgap = 2,
        ygap = 2,
        colorscale="Magma"
        )
    )
    return heatmap

@app.callback(
    [   Output(component_id='category_heatmap2', component_property='options'),
        Output(component_id='category_heatmap2', component_property='value'),
    ],
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='metric_selection', component_property='value'),
    ],
)
def update_output(model_name, metric_name):
    df = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name)]
    list_of_categories = sorted(df['task_category'].unique().tolist())
    return [{'label': str(i), 'value': str(i)} for i in list_of_categories], list_of_categories[0]

@app.callback(
    Output(component_id='heatmap2', component_property='figure'),
    [   
        Input(component_id='model_selection', component_property='value'), 
        Input(component_id='metric_selection', component_property='value'),
        Input(component_id='category_heatmap2', component_property='value'),
    ],
)
def update_output(model_name, metric_name, category):
    df = df_data[(df_data['hf_model_name'] == model_name) & (df_data['metric_names'] == metric_name) & (df_data['task_category'] == category)]
    df = df.sort_values('full_language', ascending=False)
    heatmap2 = go.Figure(
        layout=go.Layout(
        height=1000,
        )
    )
    if category != '':
        heatmap2.add_trace(
            go.Heatmap(
            name="Number",
            y = df['full_language'].tolist(),
            x = df['layer'].tolist(),
            z = np.array(df['metric_scores'].tolist()),
            xgap = 2,
            ygap = 2,
            colorscale="Magma"
            )
        )
    heatmap2.update_layout(
        xaxis = dict(tickmode = 'linear', dtick = 1),
    )
    return heatmap2


@app.callback(
    [   Output(component_id='language_heatmap3', component_property='options'),
        Output(component_id='language_heatmap3', component_property='value'),
    ],
    [   
        Input(component_id='model_heatmap3', component_property='value'), 
    ],
)
def update_output(model_name):
    df = df_data[(df_data['hf_model_name'] == model_name)]
    list_of_languages = sorted(df['full_language'].unique().tolist())
    return [{'label': str(i), 'value': str(i)} for i in list_of_languages], list_of_languages[0]


@app.callback(
    [   Output(component_id='classifier_heatmap3', component_property='options'),
        Output(component_id='classifier_heatmap3', component_property='value'),
    ],
    [   
        Input(component_id='model_heatmap3', component_property='value'), 
        Input(component_id='language_heatmap3', component_property='value'), 

    ],
)
def update_output(model_name, lang_name):
    df = df_data[(df_data['hf_model_name'] == model_name) & (df_data['full_language'] == lang_name)]
    list_of_classifiers = sorted(df['classifier_name'].unique().tolist())
    return [{'label': str(i), 'value': str(i)} for i in list_of_classifiers], list_of_classifiers[0]

@app.callback(
    [   Output(component_id='metric_heatmap3', component_property='options'),
        Output(component_id='metric_heatmap3', component_property='value'),
    ],
    [   
        Input(component_id='model_heatmap3', component_property='value'), 
        Input(component_id='language_heatmap3', component_property='value'), 
        Input(component_id='classifier_heatmap3', component_property='value'), 
    ],
)
def update_output(model_name, lang_name, classifier_name):
    df = df_data[(df_data['hf_model_name'] == model_name) & (df_data['full_language'] == lang_name) & (df_data['classifier_name'] == classifier_name)]
    list_of_metrics = sorted(df['metric_names'].unique().tolist())
    return [{'label': str(i), 'value': str(i)} for i in list_of_metrics], list_of_metrics[0]


@app.callback(
    Output(component_id='heatmap3', component_property='figure'),
    [   
        Input(component_id='model_heatmap3', component_property='value'), 
        Input(component_id='language_heatmap3', component_property='value'), 
        Input(component_id='classifier_heatmap3', component_property='value'), 
        Input(component_id='metric_heatmap3', component_property='value'),
    ],
)
def update_output(model_name, lang_name, classifier_name, metric_name):
    df = df_data[(df_data['hf_model_name'] == model_name) & (df_data['classifier_name'] == classifier_name) & (df_data['full_language'] == lang_name) & (df_data['metric_names'] == metric_name)]
    heatmap3 = go.Figure(
        layout=go.Layout(
        height=1000,
        )
    )
    if model_name != '' and lang_name != '' and classifier_name != '' and metric_name != '':
        heatmap3.add_trace(
            go.Heatmap(
            name="Number",
            y = df['task_category'].tolist(),
            x = df['layer'].tolist(),
            z = np.array(df['metric_scores'].tolist()),
            xgap = 2,
            ygap = 2,
            colorscale="Magma"
            )
        )
    heatmap3.update_layout(
        xaxis = dict(tickmode = 'linear', dtick = 1),
    )
    return heatmap3


if __name__ == "__main__":  
    app.layout = html.Div(children=[
        dbc.Row([
                html.Div(children=[
                    dbc.Row([
                        dbc.Col([
                            html.H1('Probing results', style={'font-weight': 'bold'}),
                        ], width=7),

                        dbc.Col([
                            html.Div([
                                html.P('Choose a model:', style={'margin-right': '10px'}),
                                html.Div([
                                    dcc.Dropdown(
                                        id='model_selection',
                                        options = df_data['hf_model_name'].unique().tolist(),
                                        value =  df_data['hf_model_name'].unique().tolist()[0], 
                                    ),
                                ], style={'width': '16rem', 'margin-right': '1rem'}),
                                    
                                html.P('Choose a metric:', style={'margin-right': '10px'}),
                                html.Div(
                                    [
                                    dcc.Dropdown(
                                        id='metric_selection',
                                    ),
                                    ], style={'width': '12rem'}),
                                
                                html.Div([
                                    dbc.Button('Manual', id='open', n_clicks=0, color='secondary'),
                                    dbc.Modal(
                                        [
                                            dbc.ModalHeader(dbc.ModalTitle('Manual')),
                                            dbc.ModalBody(html.Div(children = [
                                                                dbc.Row([
                                                                    html.P(a) for a in manual
                                                                ])
                                                                ]
                                                )),
                                                dbc.ModalFooter(
                                                dbc.Button(
                                                    'Close', id='close', className='ms-auto', n_clicks=0,
                                                )
                                            ),
                                        ],
                                        id='modal',
                                        is_open=False,
                                    ),
                                ], style={'margin-left': '10px'}),
                            ], style={'display': 'flex', 'flex-direction': 'row'})
                        ], width=9),

                    ]),
                ]),        
                html.Hr(),
                dbc.Card([
                    dbc.CardBody(
                        dbc.Row([
                            dbc.Col([
                                html.H3('Language map',  style={'font-weight': 'bold'}),
                                html.P('Genealogical data on the languages'),
                                dcc.Dropdown(
                                    id='language_selection_map',
                                    options = languages_for_map,
                                    value = 'Abun', 
                                    multi=True
                                ),
                            ], width=5),

                            dbc.Col([
                                dcc.Graph(id='map'),
                            ], width=7),
                        ]),  
                    ),
                ],
                    body=True,
                    # color='#2D2D2D',
                    # inverse=True
                ),
                html.Div(children=[
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                    html.H4('Average values of each category', style={'font-weight': 'bold'}, className='text-center'),
                                    html.P('graph 1', style={'font-style': 'italic'}, className='text-center'),
                                    dcc.Graph(id='graph1'),
                            ]),
                        ], width=3, style={'border-right': '1px solid', 'border-color': '#cfcfcf'}),

                        dbc.Col([
                            html.Div([
                                    html.H4('Average values of language families', style={'font-weight': 'bold'}, className='text-center'),
                                    html.P('graph 2', style={'font-style': 'italic'}, className='text-center'),
                                    dcc.Graph(id='graph2'),
                                    html.P(id='note_graph2', style={'color': '#999999', 'font-style': 'italic', 'margin-top': '1.5rem'}, className='text-center')
                            ]),
                        ], width=6, style={'border-right': '1px solid', 'border-color': '#cfcfcf'}),

                        dbc.Col([
                            html.Div(id='card_info'),
                        ], width=2)     

                    ], style={'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '2rem', 'border-radius': '4px', 'margin-top': '2rem', 'background-color': '#ffffff'}),
                ]),
                html.Br(),

                html.Div(children=[
                    dbc.Row([
                        dbc.Col([
                            html.H4('Average values of each layer', style={'text-align': 'center', 'font-weight': 'bold'}),
                            html.P('graph 3', style={'font-style': 'italic'}, className='text-center'),
                            html.Div(dcc.Graph(id='graph3')),
                        ], width=6),
                
                        dbc.Col([
                                html.H4('Average values of all languages', style={'text-align': 'center', 'font-weight': 'bold'}),
                                html.P('graph 4', style={'font-style': 'italic'}, className='text-center'),
                                dcc.Graph(id='graph4'),
                        ], width=6),
                    ], style={'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '3rem', 'border-radius': '4px', 'margin-top': '2em', 'padding-bottom': '2rem'})
                                    
                ]),

                html.Br(),
                html.Div(children=[
                    dbc.Row([
                        dbc.Col([
                            html.H4('Average values of all categories\n(for each language family)', style={'text-align': 'center', 'font-weight': 'bold'}),
                            html.P('graph 5', style={'font-style': 'italic'}, className='text-center'),
                            dcc.Dropdown(
                                id='graph5_dropdown',  
                                multi=True
                            ),
                            html.Div(dcc.Graph(id='graph5')),
                        ], width=6),
                
                        dbc.Col([
                                html.H4('Values of the languages represented in the selected category', style={'text-align': 'center', 'font-weight': 'bold'}),
                                html.P('graph 6', style={'font-style': 'italic'}, className='text-center'),
                                dcc.Dropdown(id='graph6_category'),
                                dcc.Dropdown(id='graph6_languages', multi=True, style={"margin-top": "0.5rem"},), 
                                dcc.Graph(id='graph6'),
                        ], width=6),
                    ], style={'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '3rem', 'border-radius': '4px', 'margin-top': '2em', 'background-color': '#F5F9FF'})
                                    
                ]),
                html.Div(children=[
                    dbc.Row([
                        dbc.Col([
                                html.H4('Values of the categories of the selected language', style={'text-align': 'center', 'font-weight': 'bold'}),
                                html.P('graph 7', style={'font-style': 'italic'}, className='text-center'),
                                dcc.Dropdown(id='graph7_language'), 
                                dcc.Dropdown(id='graph7_categories', multi=True, style={"margin-top": "0.5rem"}),
                                dcc.Graph(id='graph7'),
                        ], width=7, style={'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '2rem', 'border-radius': '4px', 'margin-top': '2rem', 'background-color': '#ffffff'}),
                        dbc.Col([
                            html.Div(id='graph7_datasets', className='bg-transparent', style={'maxHeight': '550px', 'overflow': 'scroll', 'background-color': 'linear-gradient(to bottom, green 25%, #FFFFFF 0%)'}),
                        ], width=4, className='offset-md-1', style={'color': 'white', 'background': 'linear-gradient(#D4C0FF, #C2DAFD)', 'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '2rem', 'border-radius': '4px', 'margin-top': '2rem', 'background-color': '#ffffff'})
                    ],)
                                    
                ]),
                html.Div(children=[
                    html.H3('Statistics on language families', style={'text-align': 'center', 'padding-top': '3rem', 'font-weight': 'bold'}),
                    html.Br(),
                    dcc.Dropdown(id='family_selection'),

                    dbc.Row([
                        dbc.Col(id='card1', width=4),
                        dbc.Col(id='card2', width=4),
                        dbc.Col(id='card3', width=4),
                        ], style={'padding-top': '2rem', 'padding-bottom': '2rem'}),
                        
                    html.Div(children=[
                            dbc.Row([
                                dbc.Col([
                                        html.H4('Structure of the language family', style={'text-align': 'center', 'font-weight': 'bold'}, className='text-center'),
                                        html.P('graph 8', style={'font-style': 'italic'}, className='text-center'),
                                        dcc.Graph(id='treemap1'),
                                        ], width=4),
                                dbc.Col([
                                        html.H4('Average values by category', style={'text-align': 'center', 'font-weight': 'bold'}),
                                        html.P('graph 9', style={'font-style': 'italic'}, className='text-center'),
                                        

                                        html.Div(className='box',
                                            children=[html.Div([
                                                html.Tr([
                                                    html.Td(daq.BooleanSwitch(id='tick', on=False, color='#9B51E0')),
                                                    html.Td(html.P('Show languages', id='tick_lable')),
                                                ]),
                                                dcc.Graph(id='boxplot'),
                                            ])
                                        ]),                             
                                        
                    ], width=8),

                                ], style={'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '2rem', 'border-radius': '4px', 'margin-top': '2rem', 'background-color': '#ffffff'})
                            ], ),
    
                ]),
                dbc.Row([
                        html.P('graph 10', style={'font-style': 'italic'}, className='text-center'),
                        dbc.Col(dcc.Dropdown(id='quantity_of_languages'), width=8, style={'margin-bottom': '2rem', 'width': '30%'}),
                        html.Div(id='graphs_for_family'),
                        ], style={'margin-bottom': '3em', 'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '2rem', 'border-radius': '4px', 'margin-top': '2rem', 'margin-bottom': '2rem', 'background-color': '#ffffff'}),
                
                html.Div(children=[
                    html.H4('Average values for each "category-language" pair', style={'text-align': 'center', 'font-weight': 'bold'}, className='text-center'),
                    html.P('graph 11', style={'font-style': 'italic'}, className='text-center'),
                    dcc.Graph(id='heatmap1'),
                ], style={'align': 'center', 'justify': 'center', 'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '2rem', 'padding-bottom': '2rem', 'border-radius': '4px', 'margin-top': '2rem', 'background-color': '#ffffff'}),

                html.Div(children=[
                    html.H4('Values of all languages in which the selected category is represented', style={'text-align': 'center', 'font-weight': 'bold'}, className='text-center'),
                    html.P('graph 12', style={'font-style': 'italic'}, className='text-center'),
                    dcc.Dropdown(id='category_heatmap2'),
                    dcc.Graph(id='heatmap2')
                ], style={'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '2rem', 'border-radius': '4px', 'margin-top': '2rem', 'background-color': '#ffffff'}),

                html.Div(children=[
                    html.H4('Whatever you want', style={'text-align': 'center', 'font-weight': 'bold'}, className='text-center'),
                    html.P('graph 13', style={'font-style': 'italic'}, className='text-center'),

                    dbc.Col([
                        html.Div([
                            html.Div(
                                [        
                                dcc.Dropdown(
                                    id='model_heatmap3',
                                    options = df_data['hf_model_name'].unique().tolist(),
                                    value =  df_data['hf_model_name'].unique().tolist()[0], 
                                ),
                                ], 
                            style={'width': '12rem', 'margin-right': '1rem'}),

                            html.Div(
                                [dcc.Dropdown(id='language_heatmap3'),
                                ], 
                            style={'width': '12rem', 'margin-right': '1rem'}),

                            html.Div(
                                [dcc.Dropdown(id='classifier_heatmap3'),
                                ], 
                            style={'width': '12rem', 'margin-right': '1rem'}),

                            html.Div(
                                [dcc.Dropdown(id='metric_heatmap3'),
                                ], 
                            style={'width': '12rem', 'margin-right': '1rem'}),

                        ], style={'display': 'flex', 'flex-direction': 'row'}),
                    ], width=10),
                    
                    html.Div([
                    dcc.Graph(id='heatmap3', style={'width': '65%', 'height': '35%'})
                    ], style={'align': 'center', 'justify': 'center'}),
                ], style={'box-shadow': '0 2px 3.5px rgba(0, 0, 0, .2)', 'padding-top': '2rem', 'border-radius': '4px', 'margin-top': '2rem', 'background-color': '#ffffff'}),

                ], justify='center')
                        
        ], style={'padding': '3rem'})

    app.run_server(debug=True)
