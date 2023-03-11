"""
 <dmatrix 로 생성 된 X 데이터와 종속변수를 사용한 모델링>

 dmatrix_X_df 함수에서 반환된 독립변수 데이터 프레임으로 OLS 모델을 생성하고 
 모델과 결과 객체를 반환해주는 함수

 VIF(변수선택법) 분석을 할때 사용할 수 있음
 
 사용한 패키지 : 
     OLS 모델 생성 : sm.OLS()
"""

def modeling_dmatrix(dfy, dfX) :

    """
    * dfy : 종속변수 데이터 프레임
        - X 데이터와 같은 길이이어야 함
    * dfX : dmatrix로 생성 된 X 데이터 프레임 (상수항 포함, formula 연산 적용)
    * return : 모델을 학습하고, 모수 추정 결과 객체를 반환하는 함수
    """

    model = sm.OLS(dfy, dfX)
    result = model.fit()

    return model, result










