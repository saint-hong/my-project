"""
 <formula 연산을 적용한 독립변수 데이터 프레임 반환 함수>

patsy 패키지의 dmatrix 서브패키지를 사용하여 독립변수 데이터에 
formula의 비선형 변형 연산을 적용하고 상수항을 포함한 데이터 프레임으로 반환하는 함수

OLS 모델 생성시 from_formula 매서드 없이 사용가능 :
    sm.OLS(dfX, dfy)

VIF(변수선택법) 분석의 입력데이터로 사용할 수 있음

사용한 패키지 : 
    dmatrix 변환 : from patsy import dmatrix
"""

def dmatrix_X_df(formula, df, outlier_idx=None) :

    """
    * formula : formula 식
    * df : 상수항 미포함 X 데이터 : dfX
    * idx : outlier의 인덱스
    * return : formula 식 연산을 적용한 X 데이터 반환 : 상수항 포함
        * if : outlier 인덱스가 없으면 전체 데이터 프레임 반환
        * else : outlier 인덱스가 있으면 이것을 제외한 데이터 프레임 반환
    """

    temp_df = dmatrix(formula, df, return_type="dataframe")
    if outlier_idx == None :
        dfX_transform = temp_df
    else :
        outlier_remove_idx = list(set(range(len(df))).difference(outlier_idx))
        dfX_transform = temp_df.iloc[outlier_remove_idx]

    return dfX_transform










