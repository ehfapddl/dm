# dm
# IISE_DataMining

## Anomaly Detection NSL-KDD

### Team
역할|학번|Git|
---|---|---|
Team Leader|16102171 김영서|[yskim569](https://github.com/yskim569)
Main developer|17101934 고세윤|[17101934ksy](https://github.com/17101934ksy/IISE_DataMining)
Main developer|20102007 김수연|[ehfapddl](https://github.com/ehfapddl)

### Quickstart

구글 Colab에서 클론하기

```python
from google.colab import drive
drive.mount('/content/drive')

!ls
!pwd

%cd "/content/drive/My Drive/git"

!git clone https://github.com/17101934ksy/IISE_DataMining.git
```

### Environment(version)
Name|install|Version|
---|---|---|
python||3.7.13
numpy|!pip install numpy|1.21.6
pandas|!pip install pandas|1.3.5
matplotlib|!pip install matplotlib|3.2.2
seaborn|!pip install seaborn|0.11.2
requests|!pip install requests|2.23.0
PIL.image|!pip install image|7.1.2
sklearn|!pip install scikit-learn|1.0.2
xgboost|!pip install xgboost|0.90
lightgbm|!pip install lightgbm|2.2.3

### Package Info
NSL_Model.ipynb: 전체 데이터마이닝 과정 모듈   
main.py: 데이터마이닝 모델 재사용을 위한 모듈  
modular.py: 모델링에 사용한 외부 모듈 

```pathon
IISE_DataMining/
    __init__.py
    main.py
    modular.py
    
    data/
    save_model/
    func/
        __init__.py
        processing.py
    loader/
        __init__.py
        loader.py
    project_ipynb/
        NSL_Model.ipynb
```

### Fix Randomseed
```python
CFG = {
    'PATH': '/content/drive/MyDrive/Colab Notebooks/데이터마이닝/project',
    'SEED': 41}
def set_env(path, seed):
  np.random.seed(seed)
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  os.chdir(path)
set_env(CFG['PATH'],CFG['SEED'])
```
### Model Object
네트워크 통신의 범위는 컴퓨터를 넘어서 스마트폰과 같은 스마트 기기, 가정에서 사용하는 스마트 가전, 그리고 iot 등으로 확장되었습니다.  
이렇게 범위가 크게 증가함에 따라서 통신의 과정이 다양화 되고 복잡해 졌습니다.  
이에따라 네트워크 공격이 더욱 고도화되고 지능화 되는 등, 개인 정보 유출이나 서버 해킹 등의 위험성이 높아졌습니다.  
저희는 이상탐지를 활용해 데이터 분석을 통한 네트워크 데이터 이상탐지에 대해서 연구하여  
네트워크 침입을 탐지할 수 있는 이상탐지 모델을 만들어 보았습니다.

<img src="https://user-images.githubusercontent.com/88478829/169640650-c726ffbe-1494-430d-9da6-81ba06fbdfd7.png" width="800" height="300"/>

### Model Metrics
주제특성에 맞도록 주된 성능 평가지표는 Fbeta를 활용할 예정이고 보조수단으로 Recall을 활용하겠습니다.
<img src="https://user-images.githubusercontent.com/88478829/169639782-9fe799b4-6ce9-4154-b17f-45db8db74187.png" width="400" height="300" float="left"/>                                                                                                 <img src="https://i.stack.imgur.com/swW0x.png" width="400" height="300" float="right"/>

### Model Select
모델링 선택은 Fbata score를 사용했습니다.  
lightgbm 모델은 다양한 모델처럼 좋은 성능을 보이지만 빠르다는 장점이 있습니다.  
따라서, lightgbm을 최종 모델로 선정하여 하이퍼 파라미터를 조정하였습니다. 
Name|#Params|GridsearchCV Fbeta|Validaton Fbeta
---|---|---|---|
RandomForest|max_depth, min_samples_leaf, min_samples_split|0.9990|0.9989|
XGboost|learning_rate, gamma, max_depth|1.0|1.0|
LightGBM|learning_rate|1.0|1.0|
SVM|C, gamma, kernel|0.9914|1.0|
  
  

### Model Fine Tuning
fbeta=2로 고정한 뒤, learing_rate, max_depth를 조정해가며   
최고의 재현율과 fbeta값이 나오는 모델을 선정하는 단계입니다.
```python
lgb_score_ = []
params = []
lgb_params = {'learning_rate' : np.linspace(0.01, 0.1, 10)}
scoring = {'recall_score': make_scorer(recall_score),
          'fbeta_score': make_scorer(fbeta_score, beta=2)}
for lr in lgb_params['learning_rate']:
  for md in [md for md in range(1, 10)]:
    params.append([lr, md])
    lgb_model = lgb.LGBMClassifier(objective='binary', learning_rate=lr, n_estimators=100, subsample=0.75, 
                                colsample_bytree=0.8, tree_method='gpu_hist', random_state=CFG['SEED'],
                                max_depth=md)
    lgb_score = cross_validate(lgb_model, X_train_full, y_train_full, scoring=scoring)
    lgb_score_.append(lgb_score)
```
   
### Model Test
Fbeta는 Precision을 어느정도 반영한다는 점에서 한계가 존재합니다.   
오버피팅을 줄이는 방안으로 learning_rate 및 max_depth를 줄여서 overfitting을 줄이는 과정을 진행하였습니다.   
또한, threshold를 조정하여 Recall이 변하지 않고, FP은 증가하지만 FN을 줄일 수 있는 지점을 구하였습니다.  그 지점은 임계점이 0.22507250725072508입니다.   
최적의 모델을 후처리한 후, Fbeta, recall의 그래프, 혼동행렬 결과입니다.
```python
lgb_model = lgb.LGBMClassifier(objective='binary', learning_rate=0.01, n_estimators=100, subsample=0.75, 
                            colsample_bytree=0.8, tree_method='gpu_hist', random_state=CFG['SEED'],
                            max_depth=1)
                            ...
>>최적의 fbeta 성능: 0.988112
```
<img src="https://user-images.githubusercontent.com/105711315/169698154-00893f28-c8ec-4e8a-8575-d943ce895351.png" width="500" height="300" float="left"/>                         <img src="https://user-images.githubusercontent.com/105711315/169698129-6a0f9df7-3a4b-4357-893e-d658934efe0d.png" width="300" height="300" float="right"/>
