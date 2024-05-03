###########################################################################
# Import packages

# dash
from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx
import dash_bootstrap_components as dbc

# basic
import pandas as pd
import numpy as np
import copy

# plotly
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go

# model
import warnings
warnings.filterwarnings('ignore')

# scaling
from sklearn.preprocessing import StandardScaler
###################################################################################
# import functions

# 분석에 사용할 칼럼을 선별하여 뽑아내기 위해 적용
def process_data(df) -> pd.DataFrame:
    numeric_cols = [
        'xmeas_1', 'xmeas_10', 'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14',
        'xmeas_15', 'xmeas_16', 'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_2',
        'xmeas_20', 'xmeas_21', 'xmeas_22', 'xmeas_23', 'xmeas_24', 'xmeas_25',
        'xmeas_26', 'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_3', 'xmeas_30',
        'xmeas_31', 'xmeas_32', 'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36',
        'xmeas_37', 'xmeas_38', 'xmeas_39', 'xmeas_4', 'xmeas_40', 'xmeas_41',
        'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8', 'xmeas_9', 'xmv_1',
        'xmv_10', 'xmv_11', 'xmv_2', 'xmv_3', 'xmv_4', 'xmv_5', 'xmv_6',
        'xmv_7', 'xmv_8', 'xmv_9'
    ]
    return df[numeric_cols]

# best k select graph
def best_k_search_graph(data):

    # Inertia 값에 대한 꺾은 선 그래프 생성
    fig_inertia = make_subplots(specs=[[{"secondary_y": True}]])

    # Create and style traces
    fig_inertia.add_trace(go.Scatter(x = data['k'], y = data['inertia'], name='Elbow', 
                            line=dict(color='firebrick', width=4)), secondary_y=False)

    fig_inertia.add_trace(go.Scatter(x = data['k'], y = data['silhouette_scores'], name='Silhouette',
                            line=dict(color='royalblue', width=4)), secondary_y=True)

    # y1과 y2에 대한 이름 설정
    fig_inertia.update_yaxes(title_text="Elbow", secondary_y=False)
    fig_inertia.update_yaxes(title_text="Silhouette", secondary_y=True)

    # 배경색은 하얀색
    fig_inertia.update_layout(title='Best K',
                    xaxis_title='k',
                    template='simple_white')

    return fig_inertia

# 영향이 큰 변수 시각화
def effective_value(data):

    fig = go.Figure()

    # Create and style traces
    fig.add_trace(go.Scatter(x=data['name'], y=data['diff'], name='difference',
                            line=dict(color='firebrick', width=4)))

    fig.add_trace(go.Bar(x=data['name'], y=data['normal'], name='normal', marker = {'color':'royalblue'}))

    fig.add_trace(go.Bar(x=data['name'], y=data['anomaly'], name='anomaly', marker = {'color':'darkblue'}))

    # Edit the layout
    fig.update_layout(title='정상과 이상의 평균에 대한 차이',
                    xaxis_title='Variable',
                    yaxis_title='Means',
                    template='simple_white')

    return fig

##########################################################################
# import path
path_dir = './AD/data/'

# import data
train = pd.read_csv(path_dir + 'train.csv')
test = pd.read_csv(path_dir + 'test.csv')
df_best_k = pd.read_csv(path_dir + 'df_best_k.csv')
distance = pd.read_csv(path_dir + 'distance.csv')
test_X_scaler3_0 = pd.read_csv(path_dir + 'test_X_scaler_normal.csv')
test_X_scaler3_1 = pd.read_csv(path_dir + 'test_X_scaler_anomaly.csv')

# best_concat.labels_ 파일을 다시 읽어서 배열로 변환
labels_df = pd.read_csv(path_dir + "cluster_labels.csv")

# best_concat.cluster_centers_ 파일을 다시 읽어서 배열로 변환
clustercenter_df = pd.read_csv(path_dir + "array_data.csv")

# train_X와 test_X 데이터프레임을 연결하여 유클리드 distance를 구한 데이터 프레임
distance_concat = pd.read_csv(path_dir + "concat_distance.csv")

##########################################################################
# data handeling 
train_X = process_data(train)
test_X = process_data(test)
X = pd.concat([train_X, test_X])

# 너무 많아서 축소시켜 데이터 출력
train_X_small = train_X.head(2500)
test_X_small = test_X.head(71040)

# scaler를 적용하여 데이터 정규화
scaler_model = StandardScaler() # 적용할 함수
scaler_model.fit(train_X) # fit
X_scaler = pd.DataFrame(scaler_model.transform(X), columns = train_X.columns)

# Center 0 location
median_train = [0.25051, 0.33706, 80.107, 49.997, 
    2633.8, 25.161, 49.998, 3102.2,
    22.948, 65.788, 231.81, 3663.7,
    341.45, 94.6, 77.294, 32.188,
    8.8933, 26.383, 6.882, 18.776,
    1.6568, 32.959, 4508.8, 13.823,
    23.978, 1.2566, 18.578, 2.2633,
    4.8436, 2.2984, 0.017866, 0.8357,
    0.098577, 9.3471, 53.724, 43.828,
    26.902, 42.338, 2705.0, 75.0,
    120.4, 63.048, 41.104, 18.119,
    53.974, 24.644, 61.298, 22.217,
    40.058, 38.091, 46.53, 47.82
]

# clust center
best_cluster_centers_ = [[-0.32536515, -0.28803686, -0.5141227 , -0.01805584,  0.71806336,
-0.04684841, -0.01940836,  0.67137507, -0.01908204, -0.28406043,
-0.15471479,  0.07950243,  0.54721512, -0.22308272, -0.08147415,
0.14062752, -0.17939101,  0.41751942, -0.12196907, -0.39140498,
-0.15199841,  0.19213256,  0.324307  , -0.25820307,  0.47991298,
-0.14442526, -0.51055072, -0.23084037, -0.41076289, -0.22015   ,
-0.01957772, -0.36864083, -0.03155825,  0.12078374, -0.04625103,
0.04350337,  0.01225843,  0.00686838,  0.71554344,  0.09908263,
0.03740844,  0.08373449,  0.14041743,  0.0142415 ,  0.31993583,
-0.32542221,  0.04730883,  0.61691857, -0.36319142, -0.01805624,
-0.01940944, -0.19868119],
[ 0.33998149,  0.3009763 ,  0.53721857,  0.01886696, -0.75032082,
0.04895297,  0.02028024, -0.70153516,  0.01993926,  0.29682124,
0.16166503, -0.0830739 , -0.57179759,  0.23310424,  0.0851342 ,
-0.14694492,  0.18744976, -0.43627558,  0.12744827,  0.40898801,
0.15882662, -0.2007637 , -0.3388758 ,  0.26980229, -0.50147205,
0.15091325,  0.53348612,  0.24121038,  0.42921554,  0.23003976,
0.02045721,  0.38520124,  0.03297594, -0.12620969,  0.04832875,
-0.04545767, -0.01280911, -0.00717692, -0.7476877 , -0.1035337 ,
-0.03908894, -0.08749608, -0.14672539, -0.01488127, -0.33430826,
0.34004111, -0.04943408, -0.64463232,  0.37950702,  0.01886737,
0.02028137,  0.20760652]]

# center 최대 허용 거리
maximum_distance_for_centroid = [12.914332076471913, 12.61746641366261]

# 거리 계산 변수 목록
cols_distance = [
        'xmeas_1', 'xmeas_10', 'xmeas_11', 'xmeas_12', 'xmeas_13', 'xmeas_14',
        'xmeas_15', 'xmeas_16', 'xmeas_17', 'xmeas_18', 'xmeas_19', 'xmeas_2',
        'xmeas_20', 'xmeas_21', 'xmeas_22', 'xmeas_23', 'xmeas_24', 'xmeas_25',
        'xmeas_26', 'xmeas_27', 'xmeas_28', 'xmeas_29', 'xmeas_3', 'xmeas_30',
        'xmeas_31', 'xmeas_32', 'xmeas_33', 'xmeas_34', 'xmeas_35', 'xmeas_36',
        'xmeas_37', 'xmeas_38', 'xmeas_39', 'xmeas_4', 'xmeas_40', 'xmeas_41',
        'xmeas_5', 'xmeas_6', 'xmeas_7', 'xmeas_8', 'xmeas_9', 'xmv_1',
        'xmv_10', 'xmv_11', 'xmv_2', 'xmv_3', 'xmv_4', 'xmv_5', 'xmv_6',
        'xmv_7', 'xmv_8', 'xmv_9'
    ]
###########################################################################
# Dashboard
app = Dash(external_stylesheets=[dbc.themes.LUX])

# App layout
app.layout = dbc.Container([

    # 대시보드 제목
    html.Br(), # 띄어쓰기
    dbc.Row([
        html.H1(children='화학 공정 이상 탐지')
    ]),

    dbc.Row([
        html.Hr() # 구분선
    ]),
    ###############################################################################################################
    dbc.Row([
        html.H3(children='1. Data 파악하기')
    ]),

    # 그래프 2개
    dbc.Row([
        html.P("가지고 있는 데이터 샘플이 train: 250K, test: 710K이기 때문에 Scatter 그래프는 1/10로 줄여서 시각화 했습니다."),
        html.Br(),
        # 첫번째 그래프
        dbc.Col([
            dbc.Row([
                html.H5(children='Histogram of variables')
            ]),
            dbc.Row([
                html.Hr()
            ]),
            dbc.Row([
                html.Label('Variables')
            ]),
            dbc.Row([
                dcc.Dropdown(train_X.columns,
                    value = 'xmeas_1',
                    id = 'single_variables')
            ]),
            html.Br(),
            dbc.Row([
                dcc.Graph(figure={}, id='one-variable-graph')
            ])
        ], width = 6),

        # 두번째 그래프
        dbc.Col([
            dbc.Row([
                html.H5(children='Scatter plot of 2 continuous variables')
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Row([
                        html.Hr()
                    ]),

                    dbc.Row([
                        html.Label('Variables1'),
                    ]),
                    
                    dbc.Row([
                        dcc.Dropdown(train_X.columns,
                                     value = 'xmeas_1',
                                     id = 'scatter_variables1'),
                    ]),
                ]),

                dbc.Col([
                    dbc.Row([
                        html.Hr()
                    ]),

                    dbc.Row([
                        html.Label('Variables2')
                    ]),
                    
                    dbc.Row([
                        dcc.Dropdown(train_X.columns,
                                value = 'xmeas_2',
                                id = 'scatter_variables2'),
                    ]),
                    
                ]),
            ]),
            

            dbc.Row([
                html.Hr()
            ]),
            dbc.Row([
                dcc.Graph(figure={}, id='scatter-graph')
            ])
        ], width = 6)
    ]),
    
    # 구분선
    dbc.Row([
        html.Hr()
    ]),
    ###############################################################################################################
    # k-means에 관한 분석 결과에 대한 그래프
    dbc.Row([
        html.H3(children='2. K-means 결과')
    ]),

    dbc.Row([
        dcc.Markdown("""
            - 이상 탐지를 진행할 수 있는 여럿 모델 중 군집 분석 방법인 k-means 방법을 사용하였습니다.  
            - k를 정하기 위해 Elbow, Silhouette 방법을 각각 사용하였으며 최적의 k를 그래프로 그려 산정하였습니다.  
            - train을 활용하여 거리 계산을 진행하여 어떠한 변수가 중요하게 작용하는지 시각화 하였습니다.  
            - 오른쪽 그래프는 Standardscaler를 활용하여 표준화를 진행한 뒤에 계산하였습니다.
            """),
    ]),

    dbc.Row([
        html.Br(), # 띄어쓰기
    ]),

    # 그래프 2개
    dbc.Row([

        # 첫번째 그래프
        dbc.Col([
            dbc.Row([
                html.H5(children='Best K for 2 methods'),
            ]),
            dbc.Row([   
                html.Br(),
            ]),
            dbc.Row([
                html.Hr()
            ]),

            html.Br(),
            
            dbc.Row([
                dcc.Graph(figure=best_k_search_graph(df_best_k))
            ])
        ], width = 6),

        # 두번째 그래프
        dbc.Col([
            dbc.Row([
                html.H5(children='differenciate of normal & anomaly by test data prediction with k-means models')
            ]),

            dbc.Row([
                html.Hr()
            ]),

            html.Br(),

            dbc.Row([
                dcc.Graph(figure=effective_value(distance.sort_values('diff', ascending = False)))
            ])
        ], width = 6)
    ]),

    # 구분선
    dbc.Row([
        html.Hr()
    ]),

    # 그래프 2개
    dbc.Row([
        html.H5(children='2-1. 영향력 있는 변수 한번에 보기')
    ]),

    dbc.Row([
        html.P("영향이 있는 변수를 확인하였을 때 비교하기 위해 k-means로 testdata를 predict한 레이블에 대해 top5, bottom5를 각각 확인하였습니다."),
        html.Br(),
        # 첫번째 그래프
        dbc.Col([
            dbc.Row([
                html.H5(children='Effective Variables of Top 5')
            ]),
            dbc.Row([
                html.Hr()
            ]),
            dbc.Row([
                html.Label('Variables')
            ]),
            dbc.Row([
                dcc.Dropdown(['xmv_5', 'xmeas_16', 'xmeas_7', 'xmeas_13', 'xmv_2'],
                    value = 'xmv_5',
                    id = 'top5_variables')
            ]),
            html.Br(),
            dbc.Row([
                dcc.Graph(figure={}, id='top5_graph')
            ])
        ], width = 6),

        # 두번째 그래프
        dbc.Col([
            dbc.Row([
                html.H5(children='Effective Variables of Bottom 5')
            ]),
            dbc.Row([
                html.Hr()
            ]),
            dbc.Row([
                html.Label('Variables')
            ]),
            dbc.Row([
                dcc.Dropdown(['xmeas_40', 'xmeas_5', 'xmeas_37', 'xmeas_41', 'xmeas_9'],
                    value = 'xmeas_40',
                    id = 'bottom5_variables')
            ]),
            html.Br(),
            dbc.Row([
                dcc.Graph(figure={}, id='bottom5_graph')
            ])
        ], width = 6),
    ]),
    
    # 구분선
    dbc.Row([
        html.Hr()
    ]),
    ###############################################################################################################
    # k-means에 관한 분석 결과에 대한 그래프
    dbc.Row([
        html.H3(children='3. 값을 대입하여 직접 반영하기')
    ]),

    dbc.Row([
        dcc.Markdown("""
            - 실제 값을 대입했을 때 어떠한 값이 도출되는지 직접 확인하기 위해 적용했습니다.  
            - 왼쪽 그래프는 입력 받은 값이 어느 정도 위치에 분포해 있는지 확인할 수 있는 그래프 입니다.  
            - 오른쪽은 실제로 값을 대입하였을 때 정상인지 이상인지 판단해주는 결과를 나타냅니다.  
            - 아래의 칸에 각각 알맞은 값을 입력해 주세요.  
            - 분석해보기 버튼을 누르시면 그래프와 분석 결과가 출력됩니다.  
            - 왼쪽 그래프는 표준화된 값이 전체 데이터의 분포에 어디쯤에 속하는지 확인할 수 있습니다.  
            - 오른쪽 결과는 가장 가까운 center와 center의 거리를 표시해주며 정상인지 비정상인지 결과를 출력해 줍니다.
            """),
    ]),

    dbc.Row([
        html.Br(), # 띄어쓰기
    ]),

    # 입력값 52개 받아쓰기
    dbc.Row([
        html.Div([
            dbc.Col([
                dbc.Row([
                    html.Label(train_X.columns[i]),
                ]),
                dbc.Row([
                    dcc.Input(value=median_train[i], type='text', id=f'{train_X.columns[i]}_values'),
                ]),
            ], style={'display': 'inline-block', 'margin-right': '10px'}) for i in range(12)
        ]),

        html.Div([
            dbc.Col([
                dbc.Row([
                    html.Label(train_X.columns[i]),
                ]),
                dbc.Row([
                    dcc.Input(value=median_train[i], type='text', id=f'{train_X.columns[i]}_values'),
                ]),
            ], style={'display': 'inline-block', 'margin-right': '10px'}) for i in range(12, 24)
        ]),

        html.Div([
            dbc.Col([
                dbc.Row([
                    html.Label(train_X.columns[i]),
                ]),
                dbc.Row([
                    dcc.Input(value=median_train[i], type='text', id=f'{train_X.columns[i]}_values'),
                ]),
            ], style={'display': 'inline-block', 'margin-right': '10px'}) for i in range(24, 36)
        ]),

        html.Div([
            dbc.Col([
                dbc.Row([
                    html.Label(train_X.columns[i]),
                ]),
                dbc.Row([
                    dcc.Input(value=median_train[i], type='text', id=f'{train_X.columns[i]}_values'),
                ]),
            ], style={'display': 'inline-block', 'margin-right': '10px'}) for i in range(36, 48)
        ]),

        html.Div([
            dbc.Col([
                dbc.Row([
                    html.Label(train_X.columns[i]),
                ]),
                dbc.Row([
                    dcc.Input(value=median_train[i], type='text', id=f'{train_X.columns[i]}_values'),
                ]),
            ], style={'display': 'inline-block', 'margin-right': '10px'}) for i in range(48, 52)
        ]),
    ]),
    
    dbc.Row([
        html.Br(), # 띄어쓰기
    ]),

    dbc.Row([
        html.Button('분석해보기', id='drawing_location_by_writing_value', n_clicks=0)    
    ]),

    dbc.Row([
        html.Br(), # 띄어쓰기
    ]),

    dbc.Row([
        html.Hr(), # 경계선
    ]),

    # 그래프 2개
    dbc.Row([

        # 첫번째 그래프
        dbc.Col([
            dbc.Row([
                html.H5(children='location of input value')
            ]),
            dbc.Row([
                html.Hr()
            ]),
            html.Br(),
            dbc.Row([
                dcc.Graph(figure={}, id='locate_graph')
            ])
        ], width = 6),

        # 두번째 그래프
        dbc.Col([
            dbc.Row([
                html.H5(children='clusters distance')
            ]),
            dbc.Row([
                html.Hr()
            ]),
            html.Br(),
            dbc.Row([
                dcc.Graph(figure={}, id='center_graph')
            ])
        ], width = 6)
    ]),

    dbc.Row([
        html.H5(children='Predict writing value by k-means data')
    ]),

    dbc.Row([
        html.Hr()
    ]),

    html.Br(),

    dbc.Row([
        html.Br(), # 띄어쓰기
    ]),
    
    dbc.Row([

        dbc.Col([

            dbc.Row([
                dcc.Markdown("""
                        #### **The Number of Cluster Center**
                """)
            ]),

            dbc.Row([
                html.Div(id='center'),
            ]),
        ]),

        dbc.Col([
            dbc.Row([
                dcc.Markdown("""
                        #### **Distance of Center**
                """)
            ]),

            dbc.Row([
                html.Div(id='distance'),
            ]),
        ]),
    ]),

    dbc.Row([
        html.Br(), # 띄어쓰기
    ]),

    dbc.Row([
        html.Br(), # 띄어쓰기
    ]),

    dbc.Row([
        html.Div(id='anomaly'),
    ]),

])
###################################################################################
# Dashboard input & output functions

# boxplot and histogram for one variable
@callback(
    Output(component_id='one-variable-graph', component_property='figure'),
    Input(component_id='single_variables', component_property='value')
)
def one_variable_graph(col_chosen):
    # Figure  생성
    # 그래프 그리기
    fig = make_subplots(rows= 1, cols= 2)

    # 왼쪽에는 박스 플롯
    fig.add_trace(go.Box(y=test_X_small[col_chosen], name='test', marker = {'color':'firebrick'}), row = 1, col = 1)
    fig.add_trace(go.Box(y=train_X_small[col_chosen], name='train', marker = {'color':'royalblue'}), row = 1, col = 1)

    # 오른쪽에는 히스토그램
    fig.add_trace(go.Histogram(x=test_X[col_chosen], name='test', marker = {'color':'firebrick'}), row = 1, col = 2)
    fig.add_trace(go.Histogram(x=train_X[col_chosen], name='train', marker = {'color':'royalblue'}), row = 1, col = 2)

    # Edit the layout
    fig.update_layout(title='One Value'
                    )

    # 색상 조절
    fig.update_traces(#marker_color= 히스토그램 색, 
                        #marker_line_width=히스토그램 테두리 두깨,                            
                        #marker_line_color=히스토그램 테두리 색,
                        # marker_opacity = 0.9,
                        )

    return fig

# two variables of scatter plot or histogram
@callback(
    Output(component_id='scatter-graph', component_property='figure'),
    Input(component_id='scatter_variables1', component_property='value'),
    Input(component_id='scatter_variables2', component_property='value')
)
def scatter_graph(var1, var2):

    # 그래프 그리기
    fig = go.Figure()

    # 다른 변수일 경우 scatter plot
    if var1 != var2:
        fig.add_trace(go.Scatter(x = test_X_small[var1], y = test_X_small[var2], mode='markers', name='test', marker = {'color':'firebrick'}))
        fig.add_trace(go.Scatter(x = train_X_small[var1], y = train_X_small[var2], mode='markers', name='train', marker = {'color':'royalblue'}))

    # 같은 변수일 경우에는 히스토그램
    else:
        fig.add_trace(go.Histogram(x=test_X[var1], name='test', marker = {'color':'firebrick'}))
        fig.add_trace(go.Histogram(x=train_X[var1], name='train', marker = {'color':'royalblue'}))
        

    # Edit the layout
    fig.update_layout(title='Pair Value',
                       template='simple_white'
                       )

    # 색상 조절
    fig.update_traces(#marker_color= 히스토그램 색, 
                        #marker_line_width=히스토그램 테두리 두깨,                            
                        #marker_line_color=히스토그램 테두리 색,
                        marker_opacity = 0.4,
                        )

    return fig

# line plot for top5 effective variable
@callback(
    Output(component_id='top5_graph', component_property='figure'),
    Input(component_id='top5_variables', component_property='value')
)
def effective_value_top_5(col_chooser):
    # top 5 그래프 그리기
    ffig = go.Figure()

    ffig.add_trace(go.Scatter(x = test_X_scaler3_0['sample'],
                            y=test_X_scaler3_0[col_chooser],
                            name=col_chooser + ' normal',
                            marker = {'color':'royalblue'}))

    ffig.add_trace(go.Scatter(x = test_X_scaler3_1['sample'],
                            y=test_X_scaler3_1[col_chooser],
                            name=col_chooser + ' anomaly',
                            marker = {'color':'firebrick'}))

    # Edit the layout
    ffig.update_layout(title='Effect top 5',
                    xaxis_title=col_chooser,
                    template='simple_white'
                    )

    return ffig

# line plot for top5 effective variable
@callback(
    Output(component_id='bottom5_graph', component_property='figure'),
    Input(component_id='bottom5_variables', component_property='value')
)
def effective_value_bottom_5(col_chooser):
    # top 5 그래프 그리기
    ffig = go.Figure()

    ffig.add_trace(go.Scatter(x = test_X_scaler3_0['sample'],
                            y=test_X_scaler3_0[col_chooser],
                            name=col_chooser + ' normal',
                            marker = {'color':'royalblue'}))

    ffig.add_trace(go.Scatter(x = test_X_scaler3_1['sample'],
                            y=test_X_scaler3_1[col_chooser],
                            name=col_chooser + ' anomaly',
                            marker = {'color':'firebrick'}))

    # Edit the layout
    ffig.update_layout(title='Effect bottom 5',
                    xaxis_title=col_chooser,
                    template='simple_white'
                    )

    return ffig

# 어느 위치에 있는지 확인하는 그래프
@callback(
    Output(component_id='locate_graph', component_property='figure'),
    Input('drawing_location_by_writing_value', 'n_clicks'),
    Input(component_id='single_variables', component_property='value'),
    [Input(component_id=f'{train_X.columns[i]}_values', component_property='value') for i in range(52)]
)
def one_variable_graph(clk ,*args):
    
    # 53개의 변수를 1변수, 1리스트로 받기
    col_chosen, variable_lists = args[0], args[1:]
    if "drawing_location_by_writing_value" == ctx.triggered_id:
        try:
            # 선택한 변수의 값 불러오기
            value = 0
            for i, c in enumerate(train_X.columns):
                if c == col_chosen:
                    value = variable_lists[i]
                    break
            
            # Figure  생성
            # 그래프 그리기
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # 전체 데이터 히스토그램
            fig.add_trace(go.Histogram(x=X[col_chosen], name='All data',
                                        marker = {'color':'royalblue'}), secondary_y=False)

            # 해당 변수가 존재하는 위치
            fig.add_trace(go.Scatter(x=[value, value], y=[-1, 10e6],
                                mode='lines', name = 'Sample', line=dict(color='firebrick', dash='dash')), secondary_y=True)

            # 어떤 변수인지 작성
            fig.update_xaxes(title_text=col_chosen)

            # Edit the layout
            fig.update_layout(title='Location of Your Values',
                            template='simple_white'
                            )
            
            # y1과 y2에 대한 이름 설정
            fig.update_yaxes(title_text="histogram", secondary_y=False)
            fig.update_yaxes(title_text="sample", secondary_y=True)

            return fig
        
        except:
            fig = go.Figure()

            # Edit the layout
            fig.update_layout(title='Location of Your Values',
                            template='simple_white'
                            )
            
            return fig

    else:
        fig = go.Figure()

        # Edit the layout
        fig.update_layout(title='Location of Your Values',
                        template='simple_white'
                        )
        
        return fig

# k-means 결과 분석
@callback(
    Output('center', 'children'),
    Output('distance', 'children'),
    Output('anomaly', 'children'),
    Input('drawing_location_by_writing_value', 'n_clicks'),
    [Input(component_id=f'{train_X.columns[i]}_values', component_property='value') for i in range(52)]
)
def result_k_means(clk, *values):

    if "drawing_location_by_writing_value" == ctx.triggered_id:
        
        # 표준화 시키기
        scaler_variable_lists = pd.DataFrame(scaler_model.transform(pd.DataFrame((values), index = train_X.columns).T), columns = train_X.columns)

        # 유클리드 거리로 centroid 계산하기
        a0 = pd.DataFrame(best_cluster_centers_, columns = train_X.columns).iloc[0, :] - scaler_variable_lists
        a0 = a0**2
        a0['dist'] = a0.apply(sum, axis = 1)

        # 유클리드 거리로 centroid 계산하기
        a1 = pd.DataFrame(best_cluster_centers_, columns = train_X.columns).iloc[1, :] - scaler_variable_lists
        a1 = a1**2
        a1['dist'] = a1.apply(sum, axis = 1)

        # cent0, cent1 대입해주기
        scaler_variable_lists['cent0'] = a0['dist']**0.5
        scaler_variable_lists['cent1'] = a1['dist']**0.5
        
        # 거리가 더 작은 값의 centroid를 선택하여 계산한 거리와 centroid 넘버를 가져옴
        scaler_variable_lists['min_d_centroid'] = np.argmin(scaler_variable_lists[[col for col in scaler_variable_lists.columns if col.startswith('cent')]], axis = 1)
        scaler_variable_lists['min_d'] = np.min(scaler_variable_lists[[col for col in scaler_variable_lists.columns if col.startswith('cent')]], axis = 1)

        # 각 centroid에 맞는 최대의 distance를 가져오기
        scaler_variable_lists['max_normal'] = scaler_variable_lists['min_d_centroid'].apply(lambda x: maximum_distance_for_centroid[x])

        # 최대 거리와 비교해서 더 크면 이상치, 같거나 작으면 정상치로 두기
        scaler_variable_lists['anomaly'] = np.where(scaler_variable_lists['min_d'] <= scaler_variable_lists['max_normal'], 'Normal', 'Anomaly')

        if scaler_variable_lists['anomaly'][0] == 'Normal':
            return html.Div(scaler_variable_lists['min_d_centroid'], style = {'font-size': '30px',
                                                                            'font-weight': 'bold',
                                                                            'text-align': 'center'}), html.Div(scaler_variable_lists['min_d'], style = {'font-size': '30px',
                                                                                                                                                        'font-weight': 'bold',
                                                                                                                                                        'text-align': 'center'}), html.Div('Normal!', style={'color': 'royalblue',
                                                                                                                                                                                                            'font-size': '100px',
                                                                                                                                                                                                            'font-weight': 'bold',
                                                                                                                                                                                                            'text-align': 'center'})
        else:
            return html.Div(scaler_variable_lists['min_d_centroid'], style = {'font-size': '30px',
                                                                            'font-weight': 'bold',
                                                                            'text-align': 'center'}), html.Div(scaler_variable_lists['min_d'], style = {'font-size': '30px',
                                                                                                                                                        'font-weight': 'bold',
                                                                                                                                                        'text-align': 'center'}), html.Div('Anomaly!', style={'color': 'firebrick',
                                                                                                                                                                                                              'font-size': '100px',
                                                                                                                                                                                                              'font-weight': 'bold',
                                                                                                                                                                                                              'text-align': 'center'})
    else:
        return '', '', ''






# histogram of distance plot
@callback(
    Output(component_id='center_graph', component_property='figure'),
    Input('drawing_location_by_writing_value', 'n_clicks'),
    [Input(component_id=f'{train_X.columns[i]}_values', component_property='value') for i in range(52)]
)
def centroid_distance_histogram(clk, *values):
    
    if "drawing_location_by_writing_value" == ctx.triggered_id:
        
        # 표준화 시키기
        scaler_variable_lists = pd.DataFrame(scaler_model.transform(pd.DataFrame((values), index = train_X.columns).T), columns = train_X.columns)
        
        # 유클리드 거리로 centroid 계산하기
        a0 = pd.DataFrame(best_cluster_centers_, columns = train_X.columns).iloc[0, :] - scaler_variable_lists
        a0 = a0**2
        a0['dist'] = a0.apply(sum, axis = 1)

        # 유클리드 거리로 centroid 계산하기
        a1 = pd.DataFrame(best_cluster_centers_, columns = train_X.columns).iloc[1, :] - scaler_variable_lists
        a1 = a1**2
        a1['dist'] = a1.apply(sum, axis = 1)

        # cent0, cent1 대입해주기
        scaler_variable_lists['cent0'] = a0['dist']**0.5
        scaler_variable_lists['cent1'] = a1['dist']**0.5
        
        # 거리가 더 작은 값의 centroid를 선택하여 계산한 거리와 centroid 넘버를 가져옴
        scaler_variable_lists['min_d_centroid'] = np.argmin(scaler_variable_lists[[col for col in scaler_variable_lists.columns if col.startswith('cent')]], axis = 1)
        scaler_variable_lists['min_d'] = np.min(scaler_variable_lists[[col for col in scaler_variable_lists.columns if col.startswith('cent')]], axis = 1)

        # 그래프 그리기
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # 전체 데이터 히스토그램
        fig.add_trace(go.Histogram(x=distance_concat['distance_to_centroid'], name='All data',
                                    marker = {'color':'royalblue'}), secondary_y=False)

        # 해당 변수가 존재하는 위치
        fig.add_trace(go.Scatter(x=[scaler_variable_lists['min_d'][0], scaler_variable_lists['min_d'][0]], y=[-1, 10e6],
                            mode='lines', name = 'Sample', line=dict(color='firebrick', dash='dash')), secondary_y=True)

        # Edit the layout
        fig.update_layout(title='Distance to Centroid Histogram')
        
        # y1과 y2에 대한 이름 설정
        fig.update_yaxes(title_text="histogram", secondary_y=False)
        fig.update_yaxes(title_text="sample", secondary_y=True)

        return fig
    
    else:
        fig = go.Figure()

        # Edit the layout
        fig.update_layout(title='Distance to Centroid Histogram')
        
        return fig


###################################################################################
# play
if __name__ == '__main__':
    app.run(debug=True)

