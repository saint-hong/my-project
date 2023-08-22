"""
<독립변수의 비선형 변형에 대한 실제값 분포와 예측값 그래프를 반환하는 함수>

독립변수와 종속값의 분포위에 비선형 변형 formula를 누적한 회귀분석 예측값 그래프 생성

비선형 변형은 다항회귀 모델 형식을 따름

차수 p값에 따라서 2차형, 3차형, 4차형 formula를 만들고 모델에 적용 :
    model_indiv : formula 단독 적용  : scale(I(CRIM**2))
    model : formual 누적 적용 : scale(CRIM) + scale(I(CRIM**2))

예측값 그래프는 formula 누적 적용한 모델만 적용

변형 formula를 단독으로 적용한 모델의 성능과 누적 적용한 모델의 성능을 각각 반환 : 
    비선형 변형을 단독으로 썼을 때와 누적해서 썼을때의 성능을 비교하기 위함
    
사용한 패키지 : 
    OLS 모델 생성 : sm.OLS.from_formula()
"""

def feature_trans(data, col, p) :

    """
    * data : 상수항 미포함, X, y 병합 데이터
    * col : 비선형 변형하려는 독립변수 이름(컬럼명)
    * p : 비선형 변형의 차수
        - 3 : 3차 다항 모델 : x + (x**2) + (x**3)
    * return :
        - 그래프 반환
        - 비선형 변형을 단독으로 사용한 모델과 누적한 모델의 성능값을 반환한다.
    """

    formula = [["I({}**{})".format(col, i+1) if i != 0 else col][0] \
               for i in range(p)]

    feature_names = []
    pred = []
    rsquared = []
    f_stats = []
    for i in range(p) :
        ## formula를 단독으로 사용한 모델 : scale(I(CRIM**2))
        model_indiv = sm.OLS.from_formula("MEDV ~ " + formula[i], data=data)
        result_indiv = model_indiv.fit()

        ## formula를 누적해서 사용한 모델 : scale(CRIM) + sale(I(CRIM**2)) + scale(I(CRIM**3))
        feature_names.append(formula[i])
        model = sm.OLS.from_formula("MEDV ~ " + "+".join(feature_names), data=data)
        result = model.fit()
        pred.append(result.predict(data))

        ## formula 단독모델과 누적 모델의 성능값을 (단독값, 누적값) 로 저장
        rsquared.append((result_indiv.rsquared, result.rsquared))
        f_stats.append((result_indiv.fvalue, result.fvalue))

    ## formula 단독 모델별 성능 데이터 프레임
    stats_df_1 = pd.DataFrame(np.vstack([[r[0] for r in rsquared],
                                         [f[0] for f in f_stats]]),
                             columns=feature_names)
    stats_df_1.index = ["R2", "f_value"]

    ## formula 누적 모델별 성능 데이터 프레임
    stats_df = pd.DataFrame(np.vstack([[r[1] for r in rsquared],
                                       [f[1] for f in f_stats]]),
                            columns=feature_names)
    stats_df.index = ["R2", "f_value"]

    ## formula 누적 모델의 예측값 데이터 프레임
    pred_df = pd.DataFrame(pred).T
    ## "I(CRIM**2)" - >"pred_I(CRIM**2)"
    new_cols = ["pred_" + str(c) for c in formula]
    pred_df.columns = new_cols
    total_df = pd.concat([pred_df, data[col]], axis=1).sort_values(col)

    ## 예측값 그래프
    plt.figure(figsize=(8, 6))
    ax = plt.subplot()

    for i in range(p) :
        temp_df = total_df[[new_cols[i], col]]
        temp_df.plot(x=col, lw=3, ax=ax)

    plt.plot(data[col], data["MEDV"], "bo", alpha=0.5)
    plt.legend(["{}={:.3f}".format(new_cols[i], rsquared[i][1]) for i in range(p)])
    plt.show() ;

    return stats_df_1, stats_df










