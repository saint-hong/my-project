"""
<OLS 모델의 회귀분석 결과 성능값을 데이터 프레임으로 반환하는 함수>

결정계수(r2), 조정결정계수(r2_adj), F-검정통계량(f_value), 
정보량규준(aic, bic) 값을 데이터 프레임으로 반환

다른 모델의 성능값 데이터 프레임을 입력받아서 현재 모델의 
성능값 데이터 프레임과 병합한 후 반환
"""

def stats_to_df(concat_df, model) :

    """
    * concat_df : 병합할 데이터 프레임 (이전 모델의 통계량 데이터 프레임)
    * model : OLS 분석의 결과인 모델 객체
        -  "모델의 이름" str로 입력
    """

    ## eval() 함수를 사용하여 모델의 이름을 객체로 변환
    obj_model = eval(model)
    temp_df = pd.DataFrame(
        {
        "r2" : obj_model.rsquared,
        "r2_adj" : obj_model.rsquared_adj,
        "f_value" : obj_model.fvalue,
        "aic" : obj_model.aic,
        "bic" : obj_model.bic
        }, columns=["r2", "r2_adj", "f_value", "aic", "bic"],
           index=["{}".format(model)])
   
    temp_df = temp_df.T
    ## concat_df와 병합
    result_df = pd.concat([concat_df, temp_df], axis=1)

    return result_df











