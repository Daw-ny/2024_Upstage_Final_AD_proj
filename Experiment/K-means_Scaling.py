
# 필요한 패키지 설치
# !pip install -r requirements.txt

# import modules
from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from pathlib import Path

# plotly
import plotly
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from plotly.subplots import make_subplots

# model
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# scaling
from sklearn.preprocessing import StandardScaler

###################################################################
# Random Seeds
RANDOM_SEED = 42
DATA_PATH = Path("../data")

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# load data
train_data = pd.read_csv(DATA_PATH / "train.csv")
test_data = pd.read_csv(DATA_PATH / "test.csv")
sample_submission = pd.read_csv(DATA_PATH / "sample_submission.csv")

###################################################################
# functions

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

# scaler를 적용하여 데이터 정규화
def scaler(train, test, scaler_type): # train, test, 사용할 scaler
    scaler = scaler_type() # 적용할 함수
    scaler.fit(train) # fit
    train_X_scaler = pd.DataFrame(scaler.transform(train), columns = train.columns) # train 적용
    test_X_scaler = pd.DataFrame(scaler.transform(test), columns = train.columns) # test 적용

    return train_X_scaler, test_X_scaler # train, test 각각 뽑아내기

# elbow method로 스코어 가져오기
def Elbow_method(n, train):

    # Inertia 값을 저장할 리스트 초기화
    inertia = []

    # 1부터 n까지의 k 값에 대해 KMeans 클러스터링 수행 및 inertia 계산
    for k in range(1, n):

        print(f'클러스터 {k}의 엘보 스코어를 계산하고 있습니다.')

        ad_kmeans = KMeans(
            n_clusters=k,    # 클러스터 수 설정
            init="random",   # 초기 중심점을 무작위로 선택
            n_init="auto",    # k-means 알고리즘 반복 실행 횟수 설정
            random_state=42
        ).fit(train)       # 데이터 적용

        # 각 k에 대한 Inertia 값 계산 및 추가
        inertia.append(ad_kmeans.inertia_)

    inertia = np.array(inertia) # np.array로 바꿔주기

    diff = inertia[:-1] - inertia[1:] # 이후와 이전 값의 차이 구하기
    best_k = np.argmax(diff) + 2 # k = 1 부터 시작하지만 차이는 k = 2부터 구할 수 있기 때문에 k = 2 적용
    print(f'Elbow Method로 확인한 결과 최적의 k는 {best_k}입니다.')

    return inertia, best_k

# Silhouette method로 best k 뽑아내기
def Silhouette_method(n, train):

    # 실루엣 분석을 사용하여 최적의 K값 탐색
    silhouette_scores = [0]

    # 1부터 n까지의 k 값에 대해 KMeans 클러스터링 수행 및 silhouette_scores 계산
    for k in range(2, n):
        
        print(f'클러스터 {k}의 실루엣 스코어를 계산하고 있습니다.')

        kmeans = KMeans(n_clusters=k,    # 클러스터 수 설정
            init="random",   # 초기 중심점을 무작위로 선택
            n_init="auto",    # k-means 알고리즘 반복 실행 횟수 설정
            random_state=42)
        
        kmeans.fit(train)
        
        # 각 스코어 계산
        score = silhouette_score(train, kmeans.labels_)

        # 값 추가
        silhouette_scores.append(score)

    print(f'silhouette method를 활용하여 나온 최적의 k는 {np.argmax(silhouette_scores) + 2} 입니다.')

    return np.array(silhouette_scores), np.argmax(silhouette_scores) + 1

# 유클리드 거리 계산하기
def Euclide_Distance(p1, p2):
    
    tmp = p1 - p2 # 두 벡터의 차이 계산
    tmp = tmp ** 2 # 제곱수 구하기
    tmp['dist'] = tmp.apply(sum, axis = 1) # 모든 값 더하기
    distance = np.array(tmp['dist'] ** 0.5) # 제곱근 계산해주기

    return distance

##################################################################################################
# k-means 최적의 k 찾기

# 필요한 열 적용하여 꺼내기
train_X = process_data(train_data)
test_X = process_data(test_data)

# Standardization 평균 0 / 분산 1
train_X_scaler, test_X_scaler = scaler(train_X, test_X, StandardScaler)

# k까지의 클러스터 중 train에 맞는 클러스터의 최적의 수를 찾기 위한 과정
inertia, best_k_elbow = Elbow_method(11, train_X_scaler)

# 실루엣 방법 적용
silhouette_scores, best_k_silhouette = Silhouette_method(11, train_X_scaler)

# Inertia 값과 클러스터 수 k를 DataFrame으로 변환
df_best_k = pd.DataFrame(dict(
    k = [x for x in range(1, 11)],  # 클러스터 수 k (1부터 10까지)
    inertia = inertia,  # 각 k에 대한 Inertia 값
    silhouette_scores = silhouette_scores,
))


##################################################################################################
# train

# # parameters
best_k = best_k_elbow # centroid의 best 개수 -> elbow, 실루엣 두 방법을 비교하여 선정한다.
abnormal = 1 # 시뮬레이션에서 이상치 판독 기준 개수

# Best k 적합하기
best = KMeans(
        n_clusters = best_k,    # 클러스터 수 설정
        init="random",   # 초기 중심점을 무작위로 선택
        n_init="auto",    # k-means 알고리즘 반복 실행 횟수 설정
        random_state=42
    ).fit(train_X_scaler)

# 각 데이터 포인트와 해당 클러스터 중심점 간의 거리 계산
distance_to_centroid = Euclide_Distance(train_X_scaler, best.cluster_centers_[best.labels_])

# 각 클러스터에 대한 최대값 계산
maximum_distance_for_centroid = [np.quantile(distance_to_centroid[best.labels_ == i], 1) for i in range(best.n_clusters)]


##################################################################################################
# inference

# 유클리드 거리로 centroid 계산하기
for i, center in enumerate(best.cluster_centers_):
    test_data[f'cent{i}'] = Euclide_Distance(test_X_scaler, best.cluster_centers_[i])

# 거리가 더 작은 값의 centroid를 선택하여 계산한 거리와 centroid 넘버를 가져옴
test_data['min_d_centroid'] = np.argmin(test_data[[col for col in test_data.columns if col.startswith('cent')]], axis = 1)
test_data['min_d'] = np.min(test_data[[col for col in test_data.columns if col.startswith('cent')]], axis = 1)

# 각 centroid에 맞는 최대의 distance를 가져오기
test_data['max_normal'] = test_data['min_d_centroid'].apply(lambda x: maximum_distance_for_centroid[x])

# 최대 거리와 비교해서 더 크면 이상치, 같거나 작으면 정상치로 두기
test_data['anomaly'] = np.where(test_data['min_d'] <= test_data['max_normal'], 0, 1)


### 이상인지 아닌지 정하기 위해 같은 SimulateRun에서 이상이 발생하면 모두 이상치로 반영
# 각 simulationRun을 맞추기 위해 이상치의 합산을 구한다.
sum_anomaly = test_data.groupby('simulationRun').sum().reset_index()[['simulationRun', 'anomaly']]

# 개수의 합이 abnormal보다 크면 1, 아니면 0
sum_anomaly['last_label'] = np.where(sum_anomaly['anomaly'] >= abnormal, 1, 0)

# abnormal에 맞춰 최종 label 생성
test_data2 = pd.merge(test_data, sum_anomaly, how = 'left', on = 'simulationRun')

# 최종적으로 값 넣어주기
sample_submission['faultNumber'] = test_data2['last_label']

# 최종적으로 만드는 제출 저장하기
sample_submission.to_csv('../data/sample_submission_k-means_scaling.csv', index = False)