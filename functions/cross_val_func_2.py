"""
<모델의 교차검증 성능값을 반환하는 함수>

train_test_split() 패키지로 교차검증

df_split_train_model 함수 : 
    train_test_split() 패키지를 사용하여 훈련, 검증 데이터로 분리
    test_size와 random_state 값 설정 가능
    훈련 데이터로 모델 학습과 모수추청
        - 상수항 미포함, X,y 병합 데이터 프레임 사용
    훈련, 검증 데이터와 모델 결과 객체 반환

calc_r2 함수 : 
    모델 객체와 검증 데이터로 모델의 성능값을 측정하고 반환하는 함수

반복문에서 df_split_train_model과 calc_r2 함수를 사용하여 교차검증 :
    train_test_split()의 random_state 값을 바꿔서 데이터 분할 비율을 다르게 함
    반복문의 반복 횟수만큼 교차검증하는 것과 같음
    train, test 교차검증 값을 데이터 프레임으로 반환
    
사용한 패키지 : 
    훈련, 검증 데이터 분리 : from sklearn.model_selection import train_test_split
"""

def df_split_train_model(data, formula, test_size, seed) :

    """
    * data : 상수항 미포함, X,y 병합 데이터
    * formula : str formula 식
    * test_size : 훈련, 검증 데이터 분할 비율
    * seed : 데이터 분할시 사용되는 random_state 난수값
    * return : 학습, 검증 데이터와 모델 결과 객체 반환
    """

    df_train, df_test = train_test_split(data, test_size=test_size,
                                         random_state=seed)

    ## 상수항 없는 데이터 프레임을 사용
    model_1, result_1 = modeling_non_const(formula, df_train)

    return df_train, df_test, result_1

def calc_r2(data, df_test, result) :

    """
    * data : 상수항 미포함, X,y 병합 데이터 
    * df_test : 검증 데이터
    * result : df_split_train_model에서 반환 된 모델 객체
        - train_test_split으로 분할한 데이터로 학습한 모델
    """

    target = data.loc[df_test.index]["MEDV"]
    predict_test = result.predict(df_test)
    RSS = ((predict_test - target)**2).sum()
    TSS = ((target - target.mean())**2).sum()
    rsquared = 1 - (RSS / TSS)
    #print(RSS, TSS)

    return rsquared

train_rsquared = []
test_rsquared = []
for seed_i in range(10) : 
    ## train_test_split 적용하기 위한 함수 호출
    (df_train, df_test, result) = df_split_train_model(
                                            df, "MEDV ~ " + formula_1, 0.2, seed_i)
    train_rsquared.append(result.rsquared)
    ## 검증 모델 성능 측정하기 위한  함수 호출
    test_rsquared.append(calc_r2(df, df_test, result))    

## 학습, 검증 모델의 성능값을 데이터 프레임으로 변환
cv_df = pd.DataFrame({"train_r2" : train_rsquared, "test_r2" : test_rsquared}, 
                     columns=["train_r2", "test_r2"])
cv_df.loc["mean"] = cv_df.mean(axis=0)
cv_df










