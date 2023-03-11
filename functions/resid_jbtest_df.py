"""
<자크베라 검정 값을 데이터 프레임으로 반환하는 함수>

잔차의 정규성을 검정하는 자크베라 테스트 값을 측정하고 데이터 프레임으로 반환

모델의 이름을 저장한 리스트로 모델별 자크베라 테스트 값을 측정 :
    eval() 함수를 사용하여 str을 객체로 변환
    
사용한 패키지 : 
    자크베라 검정 : sm.stats.jarque_bera()
"""

def resid_jbtest_df(models, q=None) : 
    
    """
    * models : 모델의 이름을 저장한 리스트
        - ["f_result_2", "f1_result_2"]
    * q : 현재까지 진행한 모형의 차수
        - 모델의 이름을 통일하기 위한 모델링의 순서값
    * return : 모형 차수별 자크베라 값의 데이터 프레임
    """
    
    ## 모델의 이름을 통일하기 위한 코드 : result_boston_1, result_boston_2 
    #models = [["result_boston" + "_" + str(i) if i != 1 else "result_boston"][0] \
          #for i in range(1, q+1)]
    
    jcb_lst = []
    for i, model in enumerate(models) : 
        eval_model = eval(model)
        jcb_stats = list(sm.stats.jarque_bera(eval_model.resid))
        jcb_lst.append(jcb_stats)
    
    jcb_df = pd.DataFrame((s for s in jcb_lst), columns=["chi2", "pvalue", "skew", "kurtosis"]).T
    jcb_df.columns = models
    jcb_df = jcb_df.apply(lambda x : round(x, 2), axis=1)
    
    return jcb_df










