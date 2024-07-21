[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/3DbKuh4a)

# 효율적인 One-Class 이상치 탐지: 화학공정 데이터를 활용한 시각화

## Team

<table>
<tr>
<td>  <div  align=center> 1 </div>  </td>
<td>  <div  align=center> 2 </div>  </td>
<td>  <div  align=center> 3 </div>  </td>
</tr>
<tr>
<td>  <div  align=center>  <b>가상민</b>  </div>  </td>
<td>  <div  align=center>  <b>신동혁</b>  </div>  </td>
<td>  <div  align=center>  <b>김도연</b>  </div>  </td>
</tr>
<tr>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/76687996/6c21c014-1e77-4ac1-89ac-72b7615c8bf5"  width="250"  height="300"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/c4cb11ba-e02f-4776-97c8-9585ae4b9f1d"  width="250"  height="300"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/3d913931-5797-4689-aea2-3ef12bc47ef0"  width="250"  height="300"/>  </td>
</tr>
<tr>
<td>  <div  align=center>  <a  href="https://github.com/3minka">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/Godjumo">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/d-yeon">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>

</tr>
</table>

<table>
<tr>
<td>  <div  align=center> 4 </div>  </td>
<td>  <div  align=center> 5 </div>  </td>
<td>  <div  align=center> 6 </div>  </td>
<td>  <div  align=center> 7 </div>  </td>
</tr>
<tr>
<td>  <div  align=center>  <b>김다운</b>  </div>  </td>
<td>  <div  align=center>  <b>서상혁</b>  </div>  </td>
<td>  <div  align=center>  <b>장호준</b>  </div>  </td>
<td>  <div  align=center>  <b>이소영</b>  </div>  </td>
</tr>
<tr>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/0f945311-9828-4e50-a60c-fc4db3fa3b9d"  width="250"  height="300"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ml-regression-01/assets/76687996/a4dbcdb5-1d28-4b91-8555-1168abffc1d0"  width="250"  height="300"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/HojunJ/conventional-repo/assets/76687996/d2bef206-7699-4028-a744-356b1950c4f1"  width="250"  height="300"/>  </td>
<td>  <img  alt="Github"  src ="https://github.com/UpstageAILab/upstage-ai-final-ir1/assets/147508048/685b52f9-872e-4456-933f-2bead5efba2b"  width="250"  height="300"/>  </td>
</tr>
<tr>
<td>  <div  align=center>  <a  href="https://github.com/Daw-ny">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/devhyuk96">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/HojunJ">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
<td>  <div  align=center>  <a  href="https://github.com/8pril">  <img  alt="Github"  src ="https://img.shields.io/badge/Github-181717.svg?&style=plastic&logo=Github&logoColor=white"/>  </div>  </td>
</tr>
</table>
  

## 0. Overview

### Environment

-   AMD Ryzen Threadripper 3960X 24-Core Processor
-   NVIDIA GeForce RTX 3090
-   CUDA Version 12.2

### Requirements

astunparse==1.6.3  
attrs==23.1.0  
brotlipy==0.7.0  
dnspython==2.4.2  
expecttest==0.1.6  
fsspec==2023.9.2  
hypothesis==6.87.1  
joblib==1.3.2  
jsonpointer==2.1  
matplotlib==3.8.2  
mkl-service==2.4.0  
nbformat==5.9.2  
pandas==2.1.4  
pathlib==1.0.1  
plotly==5.18.0  
pyarrow==14.0.2  
python-dateutil==2.8.2  
python-etcd==0.4.5  
scikit-learn==1.3.2  
scipy==1.11.4  
sortedcontainers==2.4.0  
threadpoolctl==3.2.0  
triton==2.1.0  
types-dataclasses==0.6.6  
tzdata==2023.4   

## 1. Competiton Info

### Overview

24시간 내내 운영되는 화학 공정은 이상이 발생하면 막대한 금전적 피해를 입을 수 있습니다. 공정 상태를 예측하고 대비책을 마련하는 것이 중요한 과제인데, 이를 위해서는 공정 데이터를 이해하고 이상 징후를 파악하는 것이 필수적입니다.

본 대회는 화학 공정 데이터를 이용한 이상 탐지(anomaly detection)를 수행하여, 공정 데이터에서 비정상적인 동작을 탐지하는 것을 목표로 합니다. 이를 통해 공정에서 발생할 수 있는 문제를 예측하고 대비할 수 있습니다.

본 대회에서 사용되는 입력 데이터와 출력 데이터는 모두 CSV 파일 형태로 제공됩니다. 입력 데이터로는 약 25만 개의 화학 공정 데이터가 제공되며, 이에 대응하는 약 7만 2천 개의 출력 데이터가 제공됩니다.

이상 탐지를 위한 알고리즘 개발은 화학 공정 분야에서 매우 중요한 과제이며, 이를 통해 공정의 안정성을 높이고 예기치 않은 문제를 예방할 수 있다는 점에서 큰 의미가 있습니다.



## Evaluation Metric

본 대회에서는 정상과 이상에 대한 F1-Score 를 계산하여 모델의 성능을 평가합니다.

이상인 경우 : 1

정상인 경우 : 0

사용되는 정답 Label 은 위와 같으며, 실제 정답의 정상/이상과 모델의 정상/이상을 계산하여 F1 Score 를 산출합니다.

Accuracy Score 또한 리더보드에 참고용으로 제공되나, 등수 산정은 F1 Score 만을 기준으로 합니다.

![image](https://github.com/user-attachments/assets/2586565b-67e0-47c3-8305-8e4a0e808476)

## 2. Components

### Directory

![image](https://github.com/user-attachments/assets/3f7ab3f2-2468-45ec-94b5-7d366953918c)

![image](https://github.com/user-attachments/assets/9873c511-3d05-4178-90b1-e08245ee6c47)

### EDA

![image](https://github.com/user-attachments/assets/eebf5814-2382-47e3-8c44-525e43a4556f)

![image](https://github.com/user-attachments/assets/a0f8912e-5fc5-42b1-a7c2-63e53530846b)

## 3. Dimension Reduction
### 차원 축소

![image](https://github.com/user-attachments/assets/aa8f8e38-2693-4344-9cfa-64b821726f90)

![image](https://github.com/user-attachments/assets/5d59d8ff-bf1b-4e35-b45f-c168f48301da)

## 4. Modeling
### 모델링

![image](https://github.com/user-attachments/assets/e10375a2-e5b7-452d-ac6c-ac1cefbe5965)

![image](https://github.com/user-attachments/assets/a18ecdd9-b76a-437c-8e35-4ae8a1d682dc)

![image](https://github.com/user-attachments/assets/b9e3bfc1-e3bf-44f8-941b-47913792a60c)

![image](https://github.com/user-attachments/assets/1803b59b-19a1-4d46-abd0-30dfb21462bc)

![image](https://github.com/user-attachments/assets/2d56f73c-e8ae-44bd-950b-7a256db0388b)

![image](https://github.com/user-attachments/assets/55379dca-283c-408f-9cdd-a06a95e520b3)

![image](https://github.com/user-attachments/assets/634518e5-ff7d-40de-b000-cc6fa3a4133f)

![image](https://github.com/user-attachments/assets/65cae539-f8e2-45c3-9902-28be1200279d)

![image](https://github.com/user-attachments/assets/5433d0d2-05e8-4cb5-95c3-cb5f2427ed3a)

![image](https://github.com/user-attachments/assets/af7d79bb-d3df-4911-8bcb-65c992d29096)

## 5. Result
### 사후분석

![image](https://github.com/user-attachments/assets/8e4220b2-c1ef-455a-8fdd-e960712ec6ae)

### 활용방안

![image](https://github.com/user-attachments/assets/468a762d-98e8-462f-874e-00ad30006124)
- Python Dash를 활용해 InterActive Dashboard를 구성하였습니다. 임시로 전체적인 사진만 업로드 했습니다.

### Presentation
- [Google Project](https://docs.google.com/presentation/d/1yKXgkfH1cymaJseafWG27cXgZnGUap5e/edit#slide=id.g2da7f7e83b8_0_150)

## etc

### Meeting Log

- 전체적인 내용은 [진행 Notion](https://sixth-drum-9ac.notion.site/Chemical-Process-Anomaly-Detection-dcc08017db8047a3a78e97ff96f66c1e?pvs=4), [간트차트](https://sixth-drum-9ac.notion.site/Final-d590cb0c11044d83a8d2a52459747117?pvs=4)에서 확인하실 수 있습니다.
- Apr 8 ~ May 2 : Online & Offline Meeting 

### Reference

1. [화학 공정 데이터 칼럼 정보](https://chemicalada.blogspot.com/2016/02/classification-of-variables-in-chemical.html)
