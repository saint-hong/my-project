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


### 5-1. 데이터 임포트
- scikit-learn의 datasets 패키지의 boston housing data 사용
- **독립변수와 종속변수의 의미**
    - CRIM    : 도시별 1인당 범죄율
    - ZN      : 25,000평방피트 이상의 부지에 대해 구획된 주거용 토지의 비율.
    - INDUS   : 도시당 무주택 사업 면적의 비율
    - CHAS    : Charles River 더미 변수(경계가 강인 경우 1, 그렇지 않은 경우 0)
    - NOX     : 일산화질소 농도(1,000만 개당 부품 수) 지수
    - RM      : 주택당 평균 객실 수
    - AGE     : 1940년 이전에 건설된 자가 거주 단위의 비율 지수
    - DIS     : 보스턴 5개 고용 센터까지의 가중 거리 지수
    - RAD     : 방사형 고속도로 접근성 지수
    - TAX     : 10000파운드당 완전 가산세율 지수
    - PTRATIO : 마을별 교사 비율 지수
    - B       : 도시별 흑인 비율 지수
    - LSTAT   : 낮은 인구밀도 수치
    - MEDV    : 1,000달러 단위의 주택의 가격 중앙값

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
- **데이터 크기**
    - 데이터 : 506개
    - 컬럼 : 14개 (독립변수 13개, 종속변수 1개)
- **독립변수의 자료형**
    - 통계값 만으로는 자료형을 정확하게 파악하기 어렵다. 따라서 독립변수별 유니크 값의 갯수를 측정하는 것도 도움이 된다.
    - 모든 독립변수 : 실수형
    - CHAS : 범주형
    - RM, RAD : 1~24 사이의 정수값으로 이루어져 있지만 범주형 데이터로 볼 수는 없다.
        - **RM과 RAD는 카테고리형 데이터로 간주하고 처리할 수 있다.**
    - 종속변수 : 실수형
- **결측데이터 없음**

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
    - 즉 피어슨 상관계수와 함께 독립변수가 어떤 분포인지도 확인해야 상관관계가 있는지 알 수 있다.
- `선형 회귀모델과 독립변수간 상관계수의 의미`
    - 독립변수간의 상관관계는 없을 수록 좋다. 
    - 선형회귀모델의 해를 구하는 최소자승법(OLS)의 기본 전제를 따라야 하기 때문이다.
    - 풀랭크(pull-lank) 조건 : 데이터의 독립변수는 모두 독립이어야 한다.
    - 또한 독립변수간의 상관관계는 다중공선성 현상을 야기하고, 이것은 모형의 과최적화의 원인이 될 수 있다.
    - 독립변수간의 순수한 의존성(상관관계)를 파악하기 위해 부분회귀플롯이나 CCPR 플롯을 사용할 수 있다.
- `상관관계로 독립변수가 종속변수간의 어떤 의존성이 있는지 생각해 볼 수 있다.`
    - 종속변수와의 상관계수가 큰 독립변수 : RM(0.695360), LSTAT(-0.737663)
    - RM과 LSTAT가 종속변수를 결정하는 중요한 요인으로 작용하는지 더 자세한 분석이 필요해 보인다.

#### 상관관계를 통해서 알 수 있는 내용
- `종속변수와의 상관관계`
    - 음수, 양수, 0
- `데이터의 종류`
    - 실수형, 범주형 데이터 유형 파악 가능
    - 실제 데이터의 유형은 info나 describe로 파악하기 어려움
- `부분회귀 플롯과의 비교`
    - 상관관계 그래프는 다른 독립변수에 대한 의존성이 포함되어 있는 상태
    - 부분회귀 플롯이나 CCPR로 다른 독립변수에 대한 의존성을 제거한 순수한 상관관계를 파악 할 수 있다.
- `종속변수와의 상관관계 유형`
    - 선형성이면 모형에 적합
    - 비선형성이면 모형이 덜 적합
    - 이런 경우 비선형 변형이나 강제 범주형 변형으로 적합하게 할 수 있음

#### heatmap : pearson, kendall, spearman 상관관계 그래프 비교
- sns.heatmap(df.corr())
    - `pearson` : 디폴트값, 표본상관계수, 데이터와 평균의 거리를 측정하는 방식
    - `kendall` : 순위상관계수 (Rank Correlation Coefficient)
        - 단조증가 할 때 : x증가-y증가이면 부합(concordant), x증가-y감소이면 비부합(discordant)
	- 부합쌍이 비부합쌍에 비해 얼마나 많은지에 대한 비율
        - 1이면 부합쌍 100%, -1이면 비부합쌍이 100%%
    - `spearman` : 서열상관계수 (Spearmans rank corr coef)
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

```python
## 전체 독립변수의 갯수에서 3개씩 구분하여 종속변수와 짝을 맞춰주는 코드

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
- **CRIM**
    - MEDV와 음의 비선형 관계가 있는 것으로 보인다.
- **ZN**
    - 특정 범위의 실수형 값에 데이터가 분포되어 있는 형태이다. 0값에 데이터가 몰려있다.
- **INDUS**
    - MEDV와 비선형 관계가 있는 것으로 보인다.
   
```python
sns.pairplot(df[sep_cols[0]])

plt.show() ;
```
![p11.png](./images/p11.png)

#### CHAS, NOX, RM과 MEDV
- **CHAS**
    - 0, 1의 값을 이루어진 카테고리형 데이터이다.
- **NOX**
    - MEDV와 음의 비선형 상관관계가 있는 것으로 보인다.
- **RM**
    - MEDV와 양의 상관관계가 있는 것으로 보인다.
    - 데이터의 MIN, MAX 값 부근에서 선형성이 사라지는 것으로 보인다.

```python
sns.pairplot(df[sep_cols[1]])

plt.show() ; 
```
![p12.png](./images/p12.png)

#### AGE, DIS, RAD와 MEDV
- **AGE**
    - 0~100 사이의 값에 데이터가 분포되어 있다.
    - MEDV와 비선형 상관관계, DIS와 상관관계가 있는 것으로 보인다.
- **DIS**
    - MEDV와 비선형 상관관계, AGE와 상관관계가 있는 것으로 보인다.
- **RAD**
    - 특정 범위의 값에 데이터가 몰려 있는 것으로 보인다.

```python
sns.pairplot(df[sep_cols[2]])

plt.show() ; 
```
![p13.png](./images/p13.png)

#### TAX, PTRATIO, B, LSTAT와 MEDV
- **TAX**
    - 특정 범위의 값에 데이터가 몰려 있는 것으로 보인다.
- **PTRATIO**
    - MEDV와 비선형 상관관계가 있는 것으로 보인다.
- **B**
    - MEDV와 비선형 상관관계가 있는 것으로 보인다.
- **LSTAT**
    - MEDV와 비선형 상관관계가 있는 것으로 보인다.
- **MEDV**
    - 50의 값에 데이터가 몰려 있는 것으로 보인다.

```python
sns.pairplot(df[sep_cols[3]])

plt.show() ;
```
![p14.png](./images/p14.png)

### 5-8. 변수의 누적 분포
- 독립변수의 비선형 변형에 참고로 사용할 수 있다.
- **CRIM** : 지수분포의 형태에 가깝다.
- **ZN** : 균일분포의 형태에 가깝다.
- **INDUS** : 복잡한 다봉분포 형태에 가깝다.
- **CHAS** : 베르누이 분포이다.
- **NOX** : 로그 정규분포의 형태에 가깝다.
- **RM** : 정규분포의 형태에 가깝다.
- **AGE** : 베타분포의 형태에 가깝다.
- **DIS** : 로그정규분포의 형태에 가깝다.
- **RAD** : 정규분포의 형태에 가깝다.    
- **TAX** : 정규분포의 형태에 가깝다.
- **PTRATIO** : 균일분포의 형태에 가깝다. 
- **B** : 정규분포의 형태에 가깝다.
- **LSTAT** : 로그 정규분포의 형태에 가깝다.
- **MEDV** : 정규분포의 형태에 가깝다.

```python
plt.figure(figsize=(20, 20))

for i in range(14) :
    plt.subplot(4, 4, i + 1)
    col = df.columns[i]
    sns.distplot(df[[col]], rug=False, kde=True, color='k')
    plt.title("{}".format(col), fontsize=20)

plt.tight_layout()
plt.show()
```
![p15.png](./images/p15.png)


# 모델링

### 전체 모델링 현황
- `모델링 1`
    - formula : 독립변수 그대로 사용
    - 사용한 데이터 : df
- `모델링 2`
    - formula_1 : 스케일링 + C(CHAS)
    - 독립변수의 변형 적용 : CHAS
    - 사용한 데이터 : df
- `모델링 3`
    - formula_2 : 스케일링 + C(CHAS) + scale(np.log(DIS)) + scale(I(LSTAT^2))
    - 독립변수의 변형 적용 : CHAS, DIS, LSTAT
    - 사용한 데이터 : df
- `모델링 4`
    - formula_2 + 1차 아웃라이어 제거
    - 49개의 폭스 추천 아웃라이어 제거
    - 종속값이 50인 것 중에서 복구할 데이터가 없을지 생각해 보는 것도 좋을 것 같다.
    - 사용한 데이터 : df_2
- `모델링 5`
    - formula_3_2 : formula_2 + scale(I(INDUS^2)) + scale(I(NOX^2)) + C(RAD)
    - 독립변수의 변형 적용 : 상관성이 높은 INDUS, NOX, RAD, TAX
    - 사용한 데이터 : df_2
- `모델링 6`
    - formula_4 : formula_3 + scale(I(CRIM^3)) + C(np.round(RM)) + scale(I(DIS^2))
    - 독립변수의 변형 적용 : CRIM, DIS, RM
    - 사용한 데이터 : df_2
- `모델링 7`
    - formula_4 + 2차 아웃라이어 제거
    - 30개의 폭스 추천 아웃라이어 제거
    - 사용한 데이터 : df_3
- `모델링 8`
    - formula_5 : formula_4 + I(PTRATIO^2) + I(AGE^2) + I(B^2)
    - 독립변수의 변형 적용 : PTRATIO, AGE, B
    - 모든 독립변수의 비선형 변형 진행
    - 사용한 데이터 : df_3
- `모델링 9`
    - formula_5 + 아웃라이어 데이터 복구
    - 1치 아웃라이어 중 MEDV=50 인데이터에서 AGE=100, TAX=430, 660 제외한 데이터에서 5개 샘플링 후 데이터 복구
    - 사용한 데이터 : df_3_re_gen
- `모델링 10`
    - formula_5 + 3차 아웃라이어 제거
    - MEDV=50 복구 데이터가 다시 아웃라이어로 포함 됨
        - 모델링 8에서의 아웃라이어와 같다.
    - 사용한 데이터 : df_4
- `모델링 11`
    - formula_5는 모든 독립변수의 비선형 변형을 한 것
    - 아웃라이어를 3번 제거한 데이터 프레임 사용
    - 변수선택으로 특정 독립변수 제거 후 모델링 
        - VIF, corr 값을 비교하여 선택
- `모델링 12`
    - formula_6 : formula_5에서 LSTAT 독립변수를 scale(np.log(LSTAT)) 적용
    - LSTAT 독립변수의 비선형 변형을 2차형 변형에서 로그 변형으로 수정
        - 분포의 비선형 관계가 로그 변형에 좀 더 가까운 것으로 보임
    - 사용한 데이터 : df_4
- `모델링 13`
    - formula_6 사용
    - VIF, corr, ANOVA 값을 비교하여 제거할 컬럼 선택
    - PTRATIO, CRIM, NOX, ZN, DIS, AGE
    - ZN, AGE, NOX는 제거 후에도 성능이 큰 차이가 없으므로 조합하여 추가 제거
    - 성능 : ZN > AGE > (ZN, AGE) > NOX > (ZN, AGE, NOX) > DIS > CRIM > PTRATIO
- `모델링 14`
    - PCA를 사용하여 데이터의 차원 축소
    - 사용한 데이터 : df_4

## 1. 모델링 1 : m_f1
- **formula** : 독립변수 데이터와 종속변수 데이터의 변형 없이 적용
```
CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT
```

### 1-1. formula 정의

```python
formula = [col for col in df.columns if col != "MEDV"]
formula = " + ".join(formula)
formula

>>> print

'CRIM + ZN + INDUS + CHAS + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT'
```

### 1-2. formula를 사용하여 OLS 모델링
- 상수항 미포함 모델과 상수항 포함 모델 각각 2가지로 생성 및 모수추정 후 OLS report 분석

#### OLS report 분석
1) **예측 가중치 계수**
    - INDUS와 AGE의 pvalue 값이 높다. 이것은 계수의 값이 0이라는 의미로 해석할 수 있다.
2) **성능지표**
    - rsquared : 0.741
    - r2_adj   : 0.733
    - f_value  : 108.07666
    - aic      : 3025.6085
    - bic      : 3084.7801
3) **조건수(condition number)**
    - 조건수 : 15113.517599134899 (매우 높음)
    - 강한 다중 공선성 발생 : "strong multicolinearity"
    - 독립변수간의 단위 차이로 인한 스케일 문제 또는 독립변수간의 강한 상관관계가 있는 경우 발생

#### 조건수란?
- 조건수 : 데이터의 공분산 행렬의 작은 고유값, 큰 고유값의 비율값
    - 조건수가 크면 예측값의 오차도 커진다.
    - 즉 성능이 저하된다. (rsquared 값 감소)
- 조건 수 발생 이유
    - 변수들의 단위 차이
    - 다중공선성 발생 (독립변수간 강한 상관관계가 있는 경우)
- 조건수를 낮추는 방법
    - 변수들의 단위 차이로 인한 스케일의 차이가 큰 경우 : 스케일링
    - 다중 공선성이 발생하는 경우 : 변수선택(VIF), 차원축소(PCA), 정규화 등
- OLS 레포트의 오른쪽 아래에 cond. No. 값에 해당함


```python
f_trans_X = dmatrix_X_df(formula, df)
f_model, f_result = modeling_dmatrix(dfy, f_trans_X)
f_model_2, f_result_2 = modeling_non_const("MEDV ~ " + formula, df)

print(f_result_2.summary())
```
![f_report.jpg](./images/model_1/f_report.jpg)

### 1-3. 교차검증
- **과최적화 없음**
    - train data 훈련 모델과 test data 검증 모델의 성능 값에 차이가 크지 않다.

#### KFold 방식 교차검증 분석
- **과최적화 없음**
    - test score : 0.74268
    - train score : 0.71206

```python
train_s, test_s = cross_val_func(6, df, "MEDV ~" + formula)
train_s, test_s
```
![f_cv_score.jpg](./images/model_1/f_cv_score.jpg)

#### train test split 패키지를 사용한 교차검증

```python
train_rsquared = []
test_rsquared = []

for seed_i in range(10) :

    ## df, train_test_split의 test_size, random_seed 값을 파라미터로 사용
    (df_train, df_test, result) = df_split_train_model(
                                            df, "MEDV ~ " + formula, 0.2, seed_i)
    train_rsquared.append(result.rsquared)
    test_rsquared.append(calc_r2(df, df_test, result))

cv_df = pd.DataFrame({"train_r2" : train_rsquared, "test_r2" : test_rsquared},
                     columns=["train_r2", "test_r2"])
cv_df.loc["mean"] = cv_df.mean(axis=0)
cv_df
```
![f_cv_score_2.jpg](./images/model_1/f_cv_score_2.jpg)

### 1-4. 잔차의 정규성 검정 : 쟈크베라 검정
- `올바른 모형의 잔차는 정규분포를 따른다.`
    - 잔차는 정규분포를 따르는 **잡음(오차:disturbance)**을 선형변환한 것과 같으므로 정규분포를 따른다.
    - 잡음의 기대값이 x에 상관없이 0이므로, 잔차의 기대값도 x에 상관없이 0이어야 한다.
- `자크베라 검정값의 의미`
    - pvalue : 귀무가설 "이 분포는 정규분포이다."에 대한 유의확률. 0에 가까울 수록 귀무가설이 기각된다. 유의수준보다 높을 수록 귀무가설을 채택할 수 있다.
    - skew : 0에 가까울 수록 정규분포에 가깝다는 의미
    - kurtosis : 값이 작을 수록 정규분포에 가깝다는 의미

#### 자크베라 검정값 분석
- **잔차의 정규성 검증** 
    - model 1의 잔차는 정규분포가 아니다.  
    - pvalue : 0.00
    - skew : 1.52
    - kurtosis : 8.28

```python
models = ["f_result_2"]
resid_jbtest_df(models)
```
![f_jb_test.jpg](./images/model_1/f_jb_test.jpg)


### 1-5. 잔차의 정규성 검정 : QQ 플롯
- 잔차의 범위가 -10에서 20까지 넓게 분포되어 있다.
- **분포의 양 끝부분이 바깥으로 휘어져 있으므로 잔차의 분포는 정규분포라고 할 수 없다.**
    - 또한 중심분포가 직선 형태가 아닌 구부러진 형태이므로 정규분포라고 할 수 없다.

```python
plt.figure(figsize=(8, 6))
sp.stats.probplot(f_result_2.resid, plot=plt)
plt.show() ;
```
![f_qq.jpg](./images/model_1/f_qq.jpg)

## <모델링 1의 분석>
1) 독립변수를 그대로 적용한 formula를 사용한 모델1의 OLS reoprt에서 조건수가 높다는 경고가 확인 되었다. 이것은 다중공선성 현상이 발생했기 때문인데, 다중공선성은 독립변수간의 스케일이 서로 다르거나 상관성이 높은 경우 발생한다. 다중공선성은 모형의 과최적화의 원인이 될 수 있다. **따라서 다음 모델링에서 조건수를 낮추기 위해 독립변수의 스케일링을 적용 한다.**
2) 교차검증을 한 결과 학습 데이터를 사용한 경우와 검증 데이터를 사용한 경우의 성능이 크게 차이나지 않는 것으로 보아 **과최적화는 발생하지 않은 것으로 보인다.**
3) 데이터와 모델이 적합한 경우 잔차는 정규분포를 따르게 되는데, 자크베라 검정으로 정규성을 검정한 결과 잔차가 정규분포가 아닌 것으로 보인다. **데이터와 모델의 적합성을 더 향상시켜줄 필요가 있다.**
4) QQ 플롯으로 잔차의 정규분포를 측정하면 양쪽 끝이 바깥으로 꺾인 형태를 나타낸다. 또한 중심 분포 또한 직선 형태가 아닌 곡선 형태를 나타내는 것으로 보아 **잔차는 정규분포를 따르지 않는다.**
5) 1차 모델링에서 분석한 선형회귀 모형
    - $y = 36.45949 const - 0.10801 CRIM + 0.04642 ZN + 0.02056 INDUS + 2.68673 CHAS - 17.76661 NOX + 3.80987 RM + 0.00069 AGE - 1.47557 DIS + 0.30605 RAD - 0.01233 TAX - 0.95275 PTRATIO + 0.00931 B - 0.52476 LSTAT$

## 2. 모델링 2 : m_f2
- **formula_1**
```
'scale(CRIM) + scale(ZN) + scale(INDUS) + C(CHAS) + scale(NOX) + scale(RM) + scale(AGE) + scale(DIS) + scale(RAD) + scale(TAX) + scale(PTRATIO) + scale(B) + scale(LSTAT)' 
```

- **모델링 1의 OLS report에서 발생한 조건수 에러를 해결하기 위해 독립변수에 스케일링을 적용한다.**
    - 평균을 0, 표준편차를 1으로 조정한다.
- **독립변수 중 CHAS는 범주형 데이터이므로 스케일링이 아닌 범주형 처리를 적용한다.**

#### 범주형 독립변수의 범주형 처리(더미 변수화)
- `범주형 데이터를 더미변수화 하는 2가지 방법`
- 풀랭크 : 상수값 가중치를 가지지 않는 모형
    - **formula 식에 0 추가, C(CHAS) 추가**
- 축소랭크 : 상수값 가중치를 가지는 모형
    - **formula 식에 0 제외, C(CHAS) 추가**
    - 기존의 방식으로 모형을 만든 것과 같다.
- `범주형 독립변수를 더미변수화 한다고 해서 모형의 성능이 바뀌는 것은 아니다.`
    - 가중치 계수가 변화한다.

### 2-1. formula 식 설정
- formula 식에서 독립변수의 이름에 scale()을 붙여준다.
- CHAS는 C()를 붙여준다.
- 종속변수인 MEDV는 제외한다.

```python
feature_names = [ele for ele in df.columns]
feature_names = list(set(feature_names).difference(["CHAS", "MEDV"]))
formula_1 = ["scale({})".format(ele) for ele in feature_names] + ["C(CHAS)"]
formula_1

>>> print

['scale(B)',
 'scale(AGE)',
 'scale(RAD)',
 'scale(PTRATIO)',
 'scale(CRIM)',
 'scale(TAX)',
 'scale(DIS)',
 'scale(NOX)',
 'scale(ZN)',
 'scale(LSTAT)',
 'scale(RM)',
 'scale(INDUS)',
 'C(CHAS)']
```

### 2-2. CHAS 독립변수의 유니크 데이터와 분포 확인
- 유니크 데이터가 0과 1이다. 따라서 범주형 데이터로 볼 수 있다.

#### 유니크 값 확인

```python
df["CHAS"].unique(), df["CHAS"].nunique()

>>> print

(array([0., 1.]), 2)
```

#### CHAS 데이터 분포 확인

```python
plt.figure(figsize=(8, 6))
sns.distplot(df["CHAS"], rug=False, kde=True, color='k')
plt.show() ;
```
![f1_chas_dist.jpg](./images/model_2/f1_chas_dist.jpg)


### 2-3. formula_1을 사용하여 OLS 모델링

#### < OLS report 분석 >
1) **예측 가중치 계수**
    - 이전 모델의 가중치 계수의 크기가 많이 달라 졌다.
    - INDUS와 AGE의 pvalue 값은 여전히 가장 높다. 가중치에 영향을 미치지 않는 변수로 볼 수 있다.
2) **성능지표 : 이전 모델과 달라지지 않았다.**
    - rsquared : 0.741
    - r2_adj : 0.733
    - f_value : 108.07666
    - aic : 3025.6085
    - bic : 3084.7801
3) **조건수**
    - 스케일링 적용으로 다중공선성 현상이 사라져 조건수가 크게 낮아졌다.
    - 15113 -> 10

```python
f1_trans_X = dmatrix_X_df(formula_1, df)
f1_model, f1_result = modeling_dmatrix(dfy, f1_trans_X)
f1_model_2, f1_result_2 = modeling_non_const("MEDV ~ " + formula_1, df)

print(f1_result_2.summary())
```
![f1_report.jpg](./images/model_2/f1_report.jpg)

### 2-4. 성능 지표 비교
- **모델링 1과 모델의 성능이 같다.**
    - 스케일링과 CHAS 독립변수의 범주형 처리는 모델의 성능에 영향을 미치지 않았다.

```python
f1_stats_df = stats_to_df(f_stats_df, "f1_result_2")
f1_stats_df
```
![f1_stats_df.jpg](./images/model_2/f1_stats_df.jpg)

### 2-5. 교차 검증
- **모델링 1과 교차 검증 값이 같다.**

#### KFold를 사용한 교차 검증
- 과최적화 발생하지 않음
- **함수 사용**
    - cross_val_func()

```python
train_s, test_s = cross_val_func(6, df, "MEDV ~" + formula_1)
train_s, test_s
```
![f1_cv_score.jpg](./images/model_2/f1_cv_score.jpg)


#### train_test_split을 사용한 교차 검증
- 과최적화 발생하지 않음
- **함수 사용**
    - df_split_train_model()
    - calc_r2()

```python
train_rsquared = []
test_rsquared = []

for seed_i in range(10) :

    ## df, train_test_split의 test_size, random_seed 값을 파라미터로 사용
    (df_train, df_test, result) = df_split_train_model(
                                            df, "MEDV ~ " + formula_1, 0.2, seed_i)
    train_rsquared.append(result.rsquared)
    test_rsquared.append(calc_r2(df, df_test, result))

r2_df = pd.DataFrame({"train_r2" : train_rsquared, "test_r2" : test_rsquared},
                     columns=["train_r2", "test_r2"])
r2_df.loc["mean"] = cv_df.mean(axis=0)
r2_df    
```
![f1_cv_score_2.jp](./images/model_2/f1_cv_score_2.jpg)


### 2-6. 잔차의 정규성 검정 : 자크베라 검정
- **자크베라 검정의 유의 확률이 0이고, skew와 kurtosis 값이 크므로 정규분포라고 할 수 없다.**
    - pvalue   : 0.0
    - skew      : 1.52
    - kurtosis : 8.28

```python
models = ["f_result_2", "f1_result_2"]
resid_jbtest_df(models)
```
![f1_jb_test.jpg](./images/model_2/f1_jb_test.jpg)


### 2-7. 잔차의 정규성 검정 : QQ플롯
- **QQ플롯의 형태로 보아 잔차가 정규분포를 따른다고 볼 수 없다.**
    - formula_1을 사용한 모델링 2의 잔차는 모델링 1의 잔차와 분포가 같다.

```pythonplt.figure(figsize=(10, 5))
plt.subplot(121)
sp.stats.probplot(f_result_2.resid, plot=plt)
plt.title("기본 모델")

plt.subplot(122)
sp.stats.probplot(f1_result_2.resid, plot=plt)
plt.title("스케일링 적용 모델")

plt.tight_layout()
plt.show() ; 
```
![f1_qq.jpg](./images/model_2/f1_qq.jpg)

### 2-8. VIF, Correlation, ANOVA 분석
- `VIF (Variance Inflation Factor)`
    - 변수선택법 : 과최적화의 원인인 다중공선성을 없애기 위해서 다른 독립변수의 의존적인 변수를 찾는 방법
    - 다른 변수로 i번째 변수를 선형회귀 분석한 결정계수 값과 데이터의 분산, 확률변수의 분산으로 구한다.
    - VIF 값이 클 수록 다른 변수에 의존성이 높다고 볼 수 있다.
- `ANOVA ()`
    - 독립변수와 종속변수의 분산관계를 사용하여 선형회귀 모델의 성능을 평가하는 방법
    - 각 독립변수를 제거한 축소모형과 전체모형을 비교하여 독립변수별 중요도를 평가한다.
    - ANOVA 분석의 F값이 클 수록 중요도가 크다. 유의확률 값이 0에 가까울 수록 중요도가 높다.
- VIF와 corr는 과최적화가 발생하는 경우 의존적인 변수를 제거할 때 기준이 된다.
- 과최적화가 발생하지 않은 경우는 VIF, corr, ANOVA 값을 사용하여 현재 모델링에서 독립변수를 평가하는 값으로 사용할 수 있다.

#### vca 지표 분석
- vif : INDUS, NOX, RAD, TAX가 의존도가 높다는 것을 알 수 있다.
- corr : AGE, DIS, INDUS, NOX가 다른 독립변수와 상관관계가 높다는 것을 알 수 있다.
    - 한 독립변수의 상관계수값이 0.6보다 큰 것의 갯수를 의미한다.
- anova : AGE, INDUS, TAX, CRIM, B, CHAS가 중요도가 낮다는 것을 의미한다.

```python
corr_matrix, vca_df = vif_corr_anova_df(f_trans_X, f_result_2, 0.6)

vca_df
```
![f1_vca_df.jpg](./images/model_2/f1_vca_df.jpg)

- 히트맵

```python
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlGn")
plt.show() ;
```
![f1_heatmap.jpg](./images/model_2/f1_heatmap.jpg)

## <모델링 2의 분석>
1) 독립변수를 그대로 formula로 사용한 첫번째 모델에서 나타난 다중공선성 현상을 스케일링을 통해서 제거 하였다. 이로인해 조건수도 크게 낮아졌다.
2) CHAS 변수는 0과 1의 값을 갖는 카테고리 변수라는 것을 확인하였고, 범주형 처리를 적용하였다.
3) 스케일링과 범주형 처리를 적용한 결과 모델의 성능지표와 교차검증, 잔차의 정규성 검정, VCA 지표들이 변화가 거의 없었다.
4) 그러나 스케일링으로 데이터의 표준편차가 통일되면서 예측 가중치 값이 달라졌다. 부호는 그대로이나 크기가 더 커지거나 작아졌다. 즉 스케일링을 통해서 독립변수들이 종속변수에 미치는 영향력이 조정된 것으로 볼 수 있다.
5) 독립변수들과 종속변수의 순수한 상관관계를 알기위해서는 부분회귀플롯이나 CCPR 플롯을 사용하여 확인 할 수 있다.


## 3. 모델링 3 : m_f3
- **formula 변형을 위한 독립변수 비선형 변형**
     - 스케일링 + C(CHAS) 외에 다른 독립변수의 분포와 비선형성을 파악한 후 비선형 변형을 적용하여 성능을 개선
- **LSTAT의 비선형 변형**
    - LSTAT 독립변수는 종속변수와 비선형성이 뚜렷하다.
    - 2차, 3차형 변형을 적용하였을 때 예측 성능이 개선되는 것을 확인 할 수 있다.
    - 2차형 변형 적용(3차형 변형은 모형의 과최적화를 일을 킬 수 있으므로 보류하기로 함)
- **DIS의 비선형 변형**
    - DIS 독립변수 종속변수와 비선형 관계로 나타난다. 
    - 또한 DIS 독립변수의 분포도를 통해서 로그 정규분포 형태에 가깝다는 것을 알 수 있다.
    - 이러한 특징을 반영하여 로그 변환을 적용하면 모델의 성능이 조금 높아지는 것을 확인 할 수 있다. (0.061->0.085)
    - 로그 변형 적용
- **formula_2**

```
scale(CRIM) + scale(ZN) + scale(INDUS) + C(CHAS) + scale(NOX) + scale(RM) + scale(AGE) + scale(np.log(DIS)) + scale(RAD) + scale(TAX) + scale(PTRATIO) + scale(B) + scale(LSTAT) + scale(I(LSTAT**2))
```

### 3-1. 독립변수의 비선형 변형

####  LSTAT 독립변수와 종속변수의 관계
- **LSTAT 독립변수는 비선형 관계이며 2차형이 뚜렷하다.**
    - 따라서 비선형 변형으로 2차형 변형시 성능이 개선 된다.
    - 2차형 변형 적용

```python
test_indiv, test_cumula = feature_trans(df, "LSTAT", 3)
```
![f2_lstat_dist.jpg](./images/model_3/f2_lstat_dist.jpg)

- 각 차수별 변형을 누적하여 적용한 성능 값

```python
test_cumula
```
![f2_lstat_cumula.jpg](./images/model_3/f2_lstat_cumula.jpg)

#### DIS 독립변수의 변형
- **DIS의 분포 형태가 로그 정규분포 형태를 따른다는 점을 감안하여 로그 변형을 적용한다.**
- DIS도 비선형 관계를 띄며 비선형 변형 시 성능이 어느정도 개선 된다.
    - 기본 사용시 r2 : 0.062
    - 2차형 변형시 r2 : 0.095
    - 로그 변형시 r2  : 0.085

```python
test_indiv, test_cumula = feature_trans(df, "DIS", 3)
```
![f2_dis_dist.jpg](./images/model_3/f2_dis_dist.jpg)

- 각 차수별 누적 적용한 성능 값

```python
test_cumula
```
![f2_dis_cumula.jpg](./images/model_3/f2_dis_cumula.jpg)

#### DIS 의 데이터 분포
- **로그 정규분포의 형태라고 볼 수 있다.**
    - 로그 변형 한 경우 성능이 개선된다.
    - 로그 변형 적용

```python
plt.figure(figsize=(8, 6))
sns.distplot(df["DIS"], rug=False, kde=True, color='k')
plt.show()
```
![f2_dis_dist_2.jpg](./images/model_3/f2_dis_dist_2.jpg)

#### DIS 변수를 로그 변형한 경우의 모델의 성능 비교
- **로그 변형은 변형한 것만 적용한다.**
    - scale(np.log(DIS))
    - 차수 변형을 적용할 떄는 0차+1차+2차와 같이 누적하여 적용한다.

```python
dis_model, dis_result = modeling_non_const("MEDV ~ " + "DIS", df)
dis_model_log, dis_result_log = modeling_non_const("MEDV ~ " + "scale(np.log(DIS))", df)

print("DIS r2 : {}, log DIS r2 : {}".format(dis_result.rsquared, dis_result_log.rsquared)

>>> print

DIS r2 : 0.062, log DIS r2 : 0.085
```

### 3-2. formula_2를 사용하여 OLS 모델링

#### <OLS report 분석>
1) **예측 가중치 계수**
    - INDUS와 AGE의 pvaule 값이 바뀌었다.
        - INDUS : 0.738 -> 0.864
        - AGE : 0.938 -> 0.212
    - 특히 AGE의 pvalue 값이 크게 줄어 들었다. AGE는 DIS, LSTAT와 상관관계가 컸는데, DIS와 LSTAT를 변수 변형하면서 그 의존성이 줄어든 것을 보인다.
    - ZN의 가중치 크게 줄어들었는데, pvalue 값도 유의수준에 가깝게 접근했다. 즉 예측 가중치가 0에 가까워졌다는 것을 의미한다.
    - 또한 LSTAT의 비선형 변형으로 가중치가 크게 늘어났는데, 기본형과 2차형이 서로 상반된 부호로 나타난다. 종속변수에 가장 큰 영향을 미치는 변수로 생각해  볼 수 있다.
2) **성능지표 : 성능이 개선 되었다.**
    - rsquared : 0.798
    - r2_adj : 0.792
    - f_value : 138.5
    - aic : 2901
    - bic : 2965

```python
f2_trans_X = dmatrix_X_df(formula_2, df)
f2_model, f2_result = modeling_dmatrix(dfy, f2_trans_X)
f2_model_2, f2_result_2 = modeling_non_const("MEDV ~ " + formula_2, df)

print(f2_result_2.summary())
```
![f2_report.jpg](./images/model_3/f2_report.jpg)


### 3-3. 모델의 성능 지표

```python
f2_stats_df = stats_to_df(f1_stats_df, "f2_result_2")
f2_stats_df
```
![f2_stats_df.jpg](./images/model_3/f2_stats_df.jpg)


### 3-4. 교차 검증
- **모델링 3은 과최적화가 발생하지 않는다.**

#### KFold를 사용한 교차 검증

```python
train_s, test_s = cross_val_func(6, df, "MEDV ~" + formula_2)
train_s, test_s
```
![f2_cv_score.jpg](./images/model_3/f2_cv_score.jpg)


### 3-5. 잔차의 정규성 검정 : 자크베라
- **잔차의 정규성 검증 : 잔차의 정규성이 개선되었다.**
    - pvalue   : 0.0 
    - skew      : 1.52->0.78
    - kurtosis : 8.28->6.59

```python
models = ["f_result_2", "f1_result_2", "f2_result_2"]
resid_jbtest_df(models)
```
![f2_jb_test.jpg](./images/model_3/f2_jb_test.jpg)

### 3-6. 잔차의 정규성 검정 : QQ플롯
- **잔차의 분포가 선형에 가까워 졌다.**
    - 다만 중심 분포에서 벗어난 샘플들은 여전히 존재한다. 아웃라이어의 영향으로 보인다.

```python
plt.figure(figsize=(10, 6))
plt.subplot(121)
sp.stats.probplot(f1_result_2.resid, plot=plt)
plt.title("f1 모델")

plt.subplot(122)
sp.stats.probplot(f2_result_2.resid, plot=plt)
plt.title("f2 모델")

plt.tight_layout()
plt.show() ;
```
![f2_qq.jpg](./images/model_3/f2_qq.jpg)

### 3-7. VIF, Correlation, ANOVA
- **vif** : LSTAT, TAX, RAD, DIS, NOX의 의존성이 크다.
- **corr** : AGE, DIS, INDUS, NOX, TAX의 상관관계가 높다.
    - TAX의 상관관계 값이 커짐
- **anova** : AGE, INDUS, ZN, B, CHAS의 중요도가 낮다. 
    - ZN의중요도가 낮아짐, pvalue가 커진것과 같은 맥락으로 보임

#### vca 지표

```python
corr_matrix, vca_df = vif_corr_anova_df(f2_trans_X, f2_result_2, 0.6)
vca_df
```
![f2_vca_df.jpg](./images/model_3/f2_vca_df.jpg)

#### 상관관계 히트맵
- **각 독립변수 별 상관관계가 큰 독립변수 분석**
    - DIS와 LSTAT의 변수 변형으로 독립변수의 상관관계 값이 변화한 것으로 보인다. 상관관계가 0.6보다 큰 것들을 확인.
    - INDUS와 NOX는 다른 변수들과의 상관관계가 큰 것으로 보인다.
    - RAD와 TAX는 강한 상관관계를 보인다. (0.91)
    - CRIM : RAD
    - INDUS : NOX, DIS, AGE, RAD, TAX, LSTAT : NOX, DIS 가장 큼
    - NOX : INDUS, AGE, DIS, RAD, TAX : DIS 가장 큼
    - RM : LSTAT
    - AGE : INDUS, NOX, DIS, LSTAT
    - DIS : ZN, INDUS, NOX, AGE, TAX, : NOX 가장 큼
    - RAD : CRIM, INDUS, NOX, TAX : TAX 가장 큼(0.91)
    - TAX : INDUS, NOX, RAD : RAD 가장 큼(0.91)
    - LSTAT : INDUS, NOX, AGE, LSTAT^2

```python
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="YlGn")
plt.show() ; 
```
![f2_heatmap.jpg](./images/model_3/f2_heatmap.jpg)

## <모델링 3의 분석>
1) 독립변수 중 LSTAT와 DIS의 분포를 확인하니 비선형 관계가 비교적 뚜렷하게 나타났다. 이러한 비선형 관계는 모델에 적합하지 않으므로 2차형 변형을 사용하여 비선형 관계를 조절하였다. 이로인해 모델의 성능값이 조금 개선되었고, 특히 잔차의 선형성이 개선되었다.
2) 또한 DIS와 LSTAT와 강한 상관관계를 갖고 있었던 AGE의 pvalue 값이 크게 낮아졌다. DIS와 LSTAT의 비선형 변형의 영향을 받은 것으로 보인다.
3) **잔차의 QQ플롯으로 확인 해 보면 여전히 중심 분포에서 떨어진 잔차들이 나타나는데, 아웃라이어의 영향으로 보인다.** 따라서 현재 formula_2를 사용한 모델에서 쿡스 디스턴스 값을 계산하고, 폭스 추천값 기준으로 아웃라이어를 결정하고 데이터에서 제거한 후 다시 모델링 결과를 파악하고자 한다.































































