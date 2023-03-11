"""
<모델의 교차검증 r2 성능값을 반환하는 함수>

교차검증 성능의 평균과 교차검증 별 성능값 반환 :
    train, test 성능

KFold() 교차검증 용 패키지 사용 :
    random_state는 0으로 고정
    cv는 입력 받은 값으로 설정

formula를 입력받아 sm.OLS.from_formula()에 적용

검증 모델의 성능값은 모델을 사용하지 않고 직접 계산해주어야 함 
"""

def cross_val_func(cv, data, formula) : 
    
    """
    * cv : 데이터 분할 횟수
    * data : dmatrix로 변환하지 않은 데이터 프레임
        - 상수항 미포함, X,y 데이터 병합
    * formula : str으로 된 formula 식 
    * return : train score, test score
        - 평균 성능값과 교차검증 별 성능값
    """

    train_scores = np.zeros(cv)
    test_scores = np.zeros(cv)
    
    cv = KFold(n_splits=cv, shuffle=True, random_state=0)
    ## cv.split()에서 분할순서와 train, test 데이터의 인덱스를 반환
    for i, (train_idx, test_idx) in enumerate(cv.split(data)) : 
        df_train = data.iloc[train_idx]
        df_test = data.iloc[test_idx]
        
        ## OLS 모델 생성, 모수추정 : train 데이터 사용
        model_kf = sm.OLS.from_formula(formula, data=df_train)
        result_kf = model_kf.fit()
        
        ## 검증 모델의 성능값은 직접 계산 : test 데이터 사용
        pred = result_kf.predict(df_test)
        rss = ((df_test.MEDV - pred)**2).sum()
        tss = ((df_test.MEDV - df_test.MEDV.mean())**2).sum()
        rsquared = 1 - rss / tss

        train_scores[i] = result_kf.rsquared
        test_scores[i] = rsquared
    
    ## 평균 성능값과 교차검증 성능값을 리스트로 반환
    train_score = [np.mean(train_scores), train_scores]
    test_score = [np.mean(test_scores), test_scores]
    
    return train_score, test_score 










