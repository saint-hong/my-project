# 프로젝트 : 선형회귀 분석을 사용한 Boston Housing Data 집값 예측

<p align=center> <img src = "./images/boston_city.jpg" width="70%"/> </p>

## 1. 분석 개요

### 1-1. 분석 목적
1) 집값 요인에 대한 경험적 지식과 회귀 분석을 통해서 얻은 지식의 차이를 비교한다.
2) 회귀 모델을 사용하여 데이터로부터 예측값을 측정한다.
3) 선형회귀 분석 방법을 사용하여 회귀 모델의 성능을 개선하기 위한 모델링을 적용한다.

### 1-2. 분석 대상
- scikit-learn의 datasets 패키지의 boston 데이터를 사용한다.
    - UCI의 Boston Housing Data에 기초한다.

### 1-3. 분석 방법
- stats-models의 OLS 회귀분석 패지키를 사용한다.
    - 확률론적 선형회귀 모델을 기반으로 회귀분석 이론을 구현할 수 있는 다양한 서브 패키지, 클래스, 매서드를 사용할 수 있다.
    - 데이터의 독립변수별로 여러 수학연산과 비선형 변형을 쉽게 적용할 수 있는 from_formula 매서드를 모델에 직접 적용할 수 있다.
    - 선형회귀 분석 결과와 모델을 검증할 수 있는 여러가지 측정값을 쉽게 사용할 수 있다.

### 1-4. 프로젝트 순서
1) 가설 수립
    - 보스턴 지역의 주거 환경에 관한 배경지식
    - 분석 목적에 부합하는 가설 설정
2) 데이터 EDA
    - 데이터 탐색
    - 데이터 전처리
    - 독립변수 분석
3) 모델링
    - formula 설정
    - OLS model 생성, 모수추정
4) 모델 검증
    - OLS 레포트의 통계값 분석
    - 교차 검증
    - 잔차의 정규성 검정
    - VIF score, Correlation, ANOVA score
5) 모델 수정
    - 데이터 전처리
    - 독립변수의 비선형 변형
    - 수정 사항을 반영하여 다시 3), 4) 진행
6) 분석결과
    - 모델별 성능 및 통계값 비교
    - 잔차의 정규성 검정값 비교

## 2. Boston Housing Data의 배경

<p align=center><img src = "./images/boston_map_4.jpg" width="60%"/></p>

> **보스턴은** 미국 메사추세츠 주의 주 도시로 교육, 의료, 경제, 항만, 문화 산업이 발달한 미국의 대표적인 도시중 하나이다. 특히 미국의 아테네라는 수식어에 걸맞게 아름다운 보스턴 항만과 찰스강을 따라서 세계적인 수준의 대학교와 교육기관들이 자리잡고 있으며, 이들 대학교에서 운영하는 의료시설들과 다채로운 문화산업 시설들이 있어 높은 수준의 인프라 환경을 자랑하는 세계적인 도로 알려져 있다. 

<p align=center> <img src = "./images/boston_develop_map.jpg" /> </p>

> **이러한 배경에는** 19세기 후반부터 발달한 항만 산업을 토대로 경제가 규모가 커지면서 많은 사람들이 보스턴으로 모여들었고, 이후 높아진 인구밀도에 따라서 20세기 중반부터 시작되어 지속적으로 진행중인  대규모 도시재생 사업이 있었다고 볼 수 있다. 특히 보스턴의 도시재생 사업은 낙후된 도시를 새롭게 재생시키고자하는 세계 각 지역의 도시개발 사업의 모범 사례로 손 꼽히기도 한다.

<p align=center> <img src = "./images/boston_city_3.jpg" /> </p>

> **보스턴 정부는** 1950년대를 전후로 대대적인 도시재생 사업을 진행하였고 이 과정에서 도시의 환경에 관한 다양한 조사 연구가 진행 되었다. 인구밀도의 증가와 함께 도시 환경이 변화하면서 주거 문제 개선이 도시재생 사업의 주요한 해결과제가 되었다. 보스턴 정부는 오랜시간 도시 재생 사업을 전개하면서 사람, 주거, 문화, 역사, 경제를 유기적으로 연결하기 위해 노력했다. 이러한 과정에서 모범적인 사례들뿐만 아니라 여러 실패의 사례들도 발생하였는데 이를 개선해나감으로써 오늘날의 보스턴이 만들어 질 수 있었다. **이 선형회귀 분석 프로젝트의 대상으로 삼은 Boston Housing Data는 보스턴의 이러한 분위기속에서 만들어진 데이터로 보스턴 지역 주거 환경의 13가지 요인 데이터와 집값 데이터를 포함하고 있다.**

## 3. 가설 수립
1) **경험적 지식 : 집값에 가장 큰 영향을 미치는 요인은 무엇일까? (데이터의 특징 중에서 선택)**
    - CRIM : 도시별 1인당 범죄율이 높을 수록 집값은 떨어질 것이다.
    - RM : 평균 방의 갯수가 높은 집일 수록 집값은 높아질 것이다.
    - AGE : 오래 된 집일 수록 집값은 떨어질 것이다.
    - LSTAT : 인구밀도가 낮을 수록 집값은 높아질 것이다.
    - CHAS : 찰스강 변에 위치한 집일 수록 집값은 높아질 것이다.
2) **분석적 지식 : 선형회귀 모델을 통한 회귀 분석의 결과는 경험적 지식과 어떻게 다를까?**
    - 모형의 성능이 높아지는 것과 집값 요인을 분석하는 것의 의미는 무엇인가?
3) **선형회귀 모델의 회귀분석 결과를 통해서 어떤 통찰을 얻을 수 있을까?**

## 4. 모델링 과정
1) **데이터 EDA**
    - 데이터 탐색
    - 데이터 전처리
    - 독립변수 분석 및 변형
2) **모델링**
    - formula 적용
    - OLS 모델 생성
    - OLS report 분석
3) **모델 검증**
    - 모델 성능 지표 분석
    - 교차 검증
    - 잔차의 정규성 검정
    - QQ플롯
    - ANOVA, Correlation, VIF, PCA 분석
4) **검증 결과를 토대로 데이터 EDA**
    - 데이터 전처리 : 아웃라이어 제거
    - 독립변수의 비선형 변형
5) **재 모델링**
    - formula 적용
    - OLS 모델 생성
    - OLS report 분석
6) **분석 결과**
    - 모델별 성능 지표 분석
    - 모델별 잔차의 정규성 분석
    - 예측 가중치 값 비교

### 4-1. 모델링에서 사용한 전제들
- **원본 데이터를 최대한 유지하는 방향으로 모델링을 시도한다.**
- **아웃라이어 측정 및 제거는 3번으로 제한한다.**
    - 모델이 완전하지 않으므로 모델링을 할 때마다 아웃라이어가 발생한다.
- **독립변수의 비선형 변형은 2차항까지만 진행한다.**
    - 다항회귀 모델의 차수가 늘어날 수록 모델의 성능은 증가하지만 다중공선성이 커져 모델의 적합성이 떨어진다.
    - 이러한 문제를 보완하기 위해 정규화(Rdige, Lasso, ElasticNet) 모델을 사용하여 최적화 할 수 있지만 이 과정은 선형회귀 분석이 아닌 히이퍼파라미터 튜닝이 중점이 아니게 된다.
    - 그러므로 독립변수의 비선형 변형 시 차수를 제한하고 최대한 선형회귀 분석의 방법들을 사용하여 모델링을 한다.

## 5. 데이터 EDA
1) 데이터 임포트
2) 데이터 프레임 변환
3) 데이터 크기 및 독립변수의 자료형 조회
4) 데이터 통계값 조회
5) 데이터 결측값 조회
6) 상관관계 조회 : heatmap()
7) 상관관계 조회 : pairplot()
8) 부분회귀 플롯


### 5-1. 데이터 임포트
- scikit-learn의 datasets 패키지의 boston housing data 사용

```python
from sklearn.datasets import load_boston

boston = load_boston()
print(boston.DESCR)
```
![p1.png](./images/p1.png)

### 5-2. 모델링을 위한 데이터 프레임 변환
- OLS 패키지의 dmatrix 변환 방식과 formula 적용 방식에 맞게 데이터 프레임을 각각의 방식으로 구분한다.
    - 모델 검증에서 사용할 분석 방식인 VIF와 ANOVA 분석은 입력 데이터의 조건이 각각 다르다.
- **선형회귀 모델에서 상수항을 포함하는 이유는 원소가 1인 벡터를 데이터에 추가함으로써 회귀모형의 수식을 내적으로 간단하게 처리할 수 있기 때문이다.**

#### 상수항을 포함하지 않은 데이터 프레임
- patsy 패키지의 dmatrix 변환을 위한 데이터 프레임
    - 데이터에 직접 수학 연산을 적용하여 formula 를 사용하지 않고 OLS 모델을 생성할 수 있다.
    - dmatrix 변환을 하면 데이터에 상수항이 자동으로 추가된다.

```python
## 상수항 미포함 X 데이터
dfX = pd.DataFrame(boston.data, columns=boston.feature_names)

## 종속변수
dfy = pd.DataFrame(boston.target, columns=["MEDV"])

## 상수항 미포함 데이터 프레임
df = pd.concat([dfX, dfy], axis=1)
```
    
#### 상수항을 포함한 데이터 프레임
- OLS 모델의 formula 매서드를 사용하기 위한 상수항을 포함한 데이터 프레임

```python
## 상수항 포함 X 데이터
dfX_const = sm.add_constant(dfX)

## 상수항 포함 데이터 프레임
df_const = pd.concat([dfX_const, dfy], axis=1)
```

### 5-3. 데이터 크기 및 독립변수의 자료형 조회
>- **데이터 크기**
    >- 데이터 : 506개
    >- 컬럼 : 14개 (독립변수 13개, 종속변수 1개)
>- **독립변수의 자료형**
    >- 통계값 만으로는 자료형을 정확하게 파악하기 어렵다. 따라서 독립변수별 유니크 값의 갯수를 측정하는 것도 도움이 된다.
    >- 모든 독립변수 : 실수형
    >- CHAS : 범주형
    >- RM, RAD : 1~24 사이의 정수값으로 이루어져 있지만 범주형 데이터로 볼 수는 없다.
        - **RM과 RAD는 카테고리형 데이터로 간주하고 처리할 수 있다.**
    >- 종속변수 : 실수형
>- **결측데이터 없음**

#### 데이터의 정보

```python
df.info()
```
![p2.png](./images/p2.png)

### 5-4. 데이터 통계값 조회
- **독립변수의 표준편차값이 다르다는 것은 독립변수의 스케일이 다르다는 의미이다.**
    - 독립변수의 스케일이 각각 다르면 조건수(condition number)가 커져 모형의 예측이 저하된다.
    - 독립변수의 표준편차를 같도록 스케일링으로 해결 할 수 있다.
- min, max 값을 확인하여 독립변수의 유니크한 데이터의 형태를 대략적으로 확인 할 수 있다.

```python
df.describe()
```
![p3.png](./images/p3.png)
![p4.png](./images/p4.png)

#### 독립변수의 데이터 유형을 파악하기 위한 유니크 데이터의 갯수를 계산하는 함수
- CHAS는 유니크 데이터가 0, 1로 이루어져 있으므로 2개이다.
- RAD는 9개의 정수형 데이터로 이루어져 있다.
- ZN은 26개의 실수형 데이터로 이루어져 있다.

```python
def count_unique(cols) :

    """컬럼이름과 컬럼별 유니크 데이터의 갯수를 튜플로 저장하여 반환"""

    return [(col, df[col].nunique()) for col in cols]

count_unique(df.columns)    
```
![p5.png](./images/p5.png)

#### 독립변수의 표준편차 조회
- **독립변수의 표준편차가 다르다.**
    - 독립변수의 스케일이 다르면 OLS 분석에서 조건수가(condition number) 높게 나온다.

```python
df.describe().loc["std"]
```
![p6.png](./images/p6.png)

### 5-5. 결측값 조회
- **모든 독립변수에 결측 데이터 없음**
- missingno 패키지 사용
    - msno.bar() : 결측 데이터 갯수 막대 그래프
    - msno.matrix() : 결측 데이터 위치 그래프

#### bar 플롯으로 결측데이터 확인
- 모든 독립변수에 결측데이터가 없는 것을 알 수 있다.

```python
import missingno as msno

msno.bar(df)
plt.show() ;
```
![p7.png](./images/p7.png)

#### 행렬에서 결측데이터의 위치를 확인

```python
msno.matrix(df)
plt.show() ;
```
![p8.png](./images/p8.png)

### 5-6. 독립변수의 상관관계 : heatmap

#### 독립변수의 상관관계의 의미
- `pearson 상관계수의 의미`
    - 피어슨 상관계수는 선형관계의 데이터에서 유효하다.
    - 피어슨 상관계수가 0으로 나오더라도 비선형 데이터라면 상관관계가 있을 수 있다.
    - **즉 피어슨 상관계수와 함께 독립변수가 어떤 분포인지도 확인해야 상관관계가 있는지 알 수 있다.**
- `선형 회귀모델과 독립변수간 상관계수의 의미`
    - 독립변수간의 상관관계는 없을 수록 좋다. 
    - 선형회귀모델의 해를 구하는 최소자승법(OLS)의 기본 전제를 따라야 하기 때문이다.
    - 풀랭크(pull-lank) 조건 : 데이터의 독립변수는 모두 독립이어야 한다.
    - 또한 독립변수간의 상관관계는 다중공선성 현상을 야기하고, 이것은 모형의 과최적화의 원인이 될 수 있다.
    - 독립변수간의 순수한 의존성(상관관계)를 파악하기 위해 부분회귀플롯이나 CCPR 플롯을 사용할 수 있다.
- **상관관계로 독립변수가 종속변수간의 어떤 의존성이 있는지 생각해 볼 수 있다.**
    - 종속변수와의 상관계수가 큰 독립변수 : RM(0.695360), LSTAT(-0.737663)
    - RM과 LSTAT가 종속변수를 결정하는 중요한 요인으로 작용하는지 더 자세한 분석이 필요해 보인다.

#### 상관관계를 통해서 알 수 있는 내용
- 종속변수와의 상관관계
    - 음수, 양수, 0
- 데이터의 종류
    - 실수형, 범주형 데이터 유형 파악 가능
    - 실제 데이터의 유형은 info나 describe로 파악하기 어려움
- 부분회귀 플롯과의 비교
    - 상관관계 그래프는 다른 독립변수에 대한 의존성이 포함되어 있는 상태
    - 부분회귀 플롯이나 CCPR로 다른 독립변수에 대한 의존성을 제거한 순수한 상관관계를 파악 할 수 있다.
- 종속변수와의 상관관계 유형
    - 선형성이면 모형에 적합
    - 비선형성이면 모형이 덜 적합
    - 이런 경우 비선형 변형이나 강제 범주형 변형으로 적합하게 할 수 있음

#### heatmap : pearson, kendall, spearman 상관관계 그래프 비교
- sns.heatmap(df.corr())
- corr(method)
    - **pearson : 디폴트값, 표본상관계수, 데이터와 평균의 거리를 측정하는 방식**
    - kendall : 순위상관계수 (Rank Correlation Coefficient)
        - 단조증가 할 때 : x증가-y증가이면 부합(concordant), x증가-y감소이면 비부합(discordant)
	- 부합쌍이 비부합쌍에 비해 얼마나 많은지에 대한 비율
        - 1이면 부합쌍 100%, -1이면 비부합쌍이 100%%
    - spearman : 서열상관계수 (Spearmans rank corr coef)
        - 비모수적 척도
        - 데이터에서 각변수에 대한 순위를 매긴값을 기반으로 상관관계를 측정하는 방식
        - 이산확률변수에 대해서 사용가능
- **pearson, kendall, spaerman 값은 대체로 비슷하다.**

```python
corr_lst = ["pearson", "kendall", "spearman"]

plt.figure(figsize=(15, 15))
for i, corr in enumerate(corr_lst) :
    plt.subplot(2, 2, i+1)
    sns.heatmap(df.corr(method=corr), annot=True, fmt=".2f", cmap=mpl.cm.bone_r, cbar=False)
    plt.title("{} corr heatmap".format(corr), fontsize=20)

plt.show() ;
```
![p9.png](./images/p9.png)

#### heatmap : pearson, kendall, spearman 상관계수 비교
- **각 상관계수의 값은 정도의 차이는 있지만 비슷한 상관관계를 나타낸다.**

```python
corr_y = pd.DataFrame(pearson_corr.loc["MEDV"], columns=["MEDV"]).rename(columns={"MEDV":"pearson"})
corr_y["kendall"] = kendall_corr.loc["MEDV"]
corr_y["spearman"] = spearman_corr.loc["MEDV"]
corr_y
```
![p10.png](./images/p10.png)

### 5-7. 독립변수의 상관관계 : pariplot
- 각 독립변수와 종속변수의 분포도
- 전체 독립변수의 갯수에서 3개씩 구분하여 종속변수와 짝을 맞춰주는 코드

```python
sep_idx = np.arange(0, 14, 3)
sep_cols = [0] * 4

for i in range(4) :
    start_idx = sep_idx[i]
    end_idx = sep_idx[i+1]
    if i != 3 :
        sep_cols[i] = ["MEDV"] + cols[start_idx : end_idx]
    else :
        sep_cols[i] = ["MEDV"] + cols[start_idx : (end_idx+1)]

sep_cols

>>> print

[['MEDV', 'CRIM', 'ZN', 'INDUS'],
 ['MEDV', 'CHAS', 'NOX', 'RM'],
 ['MEDV', 'AGE', 'DIS', 'RAD'],
 ['MEDV', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
```

#### CRIM, ZN, INUDS와 MEDV

```python
sns.pairplot(df[sep_cols[0]])

plt.show() ;
```
![p11.png](./images/p11.png)

#### CHAS, NOX, RM과 MEDV

```python
sns.pairplot(df[sep_cols[1]])

plt.show() ; 
```
![p12.png](./images/p12.png)

#### AGE, DIS, RAD와 MEDV

```python
sns.pairplot(df[sep_cols[2]])

plt.show() ; 
```
![p13.png](./images/p13.png)

#### TAX, PTRATIO, B, LSTAT와 MEDV

```python
sns.pairplot(df[sep_cols[3]])

plt.show() ;
```
![p13.png](./images/p13.png)


### 5-8. 독리변수의 상관관계 : 부분회귀 플롯







































































































