# Kaggle Compettion Student Classification
- 케글 컴피티션 중 하나로 학생의 생활기록부 데이터를 사용하여 졸업여부에 대한 예측 모델의 성능을 테스트 한다. 학습 데이터는 포르투갈 기반의 연구소에서 제작 된 고등학생들의 생활기록부와 같은 데이터로 UCI Machine Learning Repository 가 원본 데이터이다. 다양한 학부 학위에 등록한 학생들의 고등 교육 기관에서 생선 된 여러 데이터 세트를 취합하여 만들어 졌다.   
- 이번 케글 컴피티션에 참여하게 된 것은 학습 데이터가 기계적인 과정을 거쳐 만들어졌거나 자연적으로 발생한 것이 아닌 다양한 사람들이 각자의 선택에 따라서 내린 결정들의 결과물이라는 점이 흥미로웠기 떄문이다. 인간 행위의 산물을 데이터화하고 이러한 데이터를 학습 데이터로 사용하여 머신러닝 모델로부터 미래의 어떠한 결론을 예측해보고 싶었다.
- 지금까지 예측 성능을 높이기 위한 모델링 과정에서 어떠한 경우 학습 데이터에 대한 의외로 아주 간단한 전처리나 feature engineering 만으로도 성능이 개선될 수 있다는 것을 알 수 있었다. 복잡한 하이퍼 파라미터 튜닝을 오랜시간 지속해서 얻은 모델이라고 해도 성능이 월등히 높아지지 않는 이유이기도 하다. 따라서 이번 케글 컴피티션에서는 모델링 자체에 많은 시간을 할애하기 보다 기본 모델을 하나 선택해 놓고, 다양한 전처리, 다양한 engineering을 우선 시도하여 모델 성능에 변화를 주는 어떤 데이터 상태를 찾아보고자 했다.

## Compettion Info

### 컴피티션 일정
- 6/1 부터 7/1 까지 하루에 다섯번 결과물을 제출할 수 있다. 6/16일께 참여하였다.

<img src="./images/info_1.png">

### 데이터 정보
- 컴피티션에서 제공하는 데이터는 train, test, sample_submission 3가지이다. 
- train 데이터는 76518개의 표본과 36개의 독립변수(id, target 제외)로 이루어져 있고, test 데이터는 51012개의 표본과 36개의  독립변수(feature)로 이루어져 있다. 또한 종속변수 Target은 "Graduate", "Dropout", "Enrolled" 3가지의 멀티 클래스로 이루어져 있다.
- sample_submission 파일은 test 데이터에 대한 예측값을 저장하여 제출할 때 사용하는 양식이다.

<img src=",/images/info_2.png">

### 원본 데이터
- 이 데이터는 UCI의 Machine Learning Repository의 원본 데이터를 사용했다. 또한 원본 데이터와 다르게 범주형 독립변수의 데이터가 모두 정수값으로 변환되어 있다.
- UCI의 페이지에서 각 feature에 대한 설명을 확인 할 수있다. 
   - https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success

<img src="./images/info_3.png">


## 개요

### file 1 : EDA, Preprocessor, Feature engineering, Modeling test

### file 2 : Hyper parameter tunning test

### file 3 : AutoML test

## Review



