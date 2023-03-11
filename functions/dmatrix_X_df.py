# 선형회귀 분석에 사용한 함수들

def dmatrix_X_df(formula, df, outlier_idx=None) :

    """
    <formula 연산을 적용한 X데이터 반환 함수>

    * formula : formula 식
    * df : 상수항 미포함 X 데이터 : dfX
    * idx : outlier의 인덱스
    * return : formula 식 연산을 적용한 X 데이터 반환 : 상수항 포함
       * if : outlier 인덱스가 없으면 전체 데이터 프레임 반환
       * else : outlier 인덱스가 있으면 이것을 제외한 데이터 프레임 반환

    * VIF(변수선택법) 측정을 하기 위해서 사용
    """

    temp_df = dmatrix(formula, df, return_type="dataframe")

    if outlier_idx == None :
        dfX_transform = temp_df

    else :
        outlier_remove_idx = list(set(range(len(df))).difference(outlier_idx))
        dfX_transform = temp_df.iloc[outlier_remove_idx]

    return dfX_transform







