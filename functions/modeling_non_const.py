"""
<상수항 없는 데이터 프레임을 사용한 모델링 함수>

OLS 패키지의 from_formula 매서드를 사용하여 OLS 모델을 생성하기 위한 함수

X와 y 데이터가 포함된 데이터 프레임을 사용 :
    단, 데이터에 상수항이 없어야 함

ANOVA(분산분석)에서 사용할 수 있음

사용한 패키지 : 
    OLS 모델 생성 : sm.OLS.from_formula()
"""

def modeling_non_const(formula, data) :

    """
    * formula : "MEDV ~ scale(CRIM) + scale(ZN) + ..." : str
    * data : 상수항이 없는 X 데이터와 y 데이터로 결합된 데이터 프레임
    * return : 모델의 모수추정 결과 반환
    """

    model = sm.OLS.from_formula(formula, data=data)
    result = model.fit()

    return model, result










