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
<td>  <img  alt="Github"  src ="https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/85b14aed-b780-4d9d-b51e-5c497b7c4220"  width="250"  height="300"/>  </td>
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

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/d4cfde14-b602-4862-b536-37a4f6970be7)

## 2. Components

### Directory

![image](https://github.com/UpstageAILab/upstage-cv-classification-cv1/assets/76687996/17569632-122c-4b30-93d1-3c08717d32e1)

## 3. Strategy

### Dataset overview
![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/bfa1e3ff-c8a4-48b7-997e-3dfca4ccbcb8)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/e2bfc1a5-7c2a-41c0-9d8a-dcee80e0d065)

### EDA

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/7ee06bc9-54e8-48a4-ba76-623861b4c3bd)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/41319b9c-6573-41c7-81cc-9a51569d3c8e)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/6994102c-db13-48c8-bdc9-f86bf244e165)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/bee81b78-d281-4f6b-a846-bd272712aa07)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/cd52c5a6-e273-4ec6-ac37-dec037a3f685)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/c1d48682-8fe3-4e62-ac68-3e42d926b05e)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/b06954e8-f956-445e-b7f2-ee61afe539c7)


## Modeling

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/2398e6b9-de3b-49cb-8bda-bf8c15ef9723)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/90a9a878-c6f0-4478-8cc2-eeec315d5b12)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/b71452ae-5d61-4eb6-9f41-b00b6591ed7f)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/a49771cc-ee62-4e91-82bd-8aae5888a299)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/69c928be-6825-49fc-a873-3ebe28db152f)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/73134fac-a5ff-4066-97b4-0e6d11f4dd0c)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/7c929557-07d4-4849-be1e-32a69e32bf41)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/474338b4-5978-436c-b3d4-f3d2b9d3ac04)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/f3fd9bb2-216c-4d5d-8fb7-b8af1a3e07e0)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/c304326a-9757-46a5-a71f-bdd9f9ec26e1)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/88e45cee-30cd-4b8b-b61f-3d4c536077b2)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/248030e5-2fce-41dd-ac98-c6578a94bfe2)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/393e5314-4f58-4d3c-bdb3-5e30840e660b)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/e1465d2c-568e-45ac-9f5f-bd37ecf55445)

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/86c09442-480f-4695-8ecf-993ffa262672)

## 5. Result

### Leader Board - 1st

![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/3093d018-13da-4665-9eb4-e701058b2f7d)

### 활용 방안
![image](https://github.com/Daw-ny/2023_Dacon_pred_temp/assets/76687996/02fcf1ce-ae8d-47d1-841e-6efeb1fdcaab)
- Python Dash를 활용해 InterActive Dashboard를 구성하였습니다. 임시로 전체적인 사진만 

### Presentation
- [Google Project](https://docs.google.com/presentation/d/1MJICWO11-tY88Fn-WPqpa4e8h4p83jTm/edit#slide=id.g2c4e1232ee9_0_11)

## etc

### Meeting Log

- 전체적인 내용은 [진행 Notion](https://sixth-drum-9ac.notion.site/Chemical-Process-Anomaly-Detection-dcc08017db8047a3a78e97ff96f66c1e?pvs=4), [간트차트](https://sixth-drum-9ac.notion.site/Final-d590cb0c11044d83a8d2a52459747117?pvs=4)에서 확인하실 수 있습니다.
- Apr 8 ~ May 2 : Online & Offline Meeting 

### Reference

1. [화학 공정 데이터 칼럼 정보](https://chemicalada.blogspot.com/2016/02/classification-of-variables-in-chemical.html)
