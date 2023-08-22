"""
<VIF 값 데이터 프레임을 반환하는 함수>

dmatrix_X_df로 변환한 X 데이터를 사용하여 VIF 값을 측정 :
    X 데이터에는 formula 연산이 적용되어 있음
    상수항이 포함되어 있음

데이터 프레임 반환

사용한 패키지 : 
     from statsmodels.stats.outliers_influence import variance_inflation_factor
"""

def calc_vif(data_X) : 
    
    """    
    * data_X : dmatrix에서 반환 된 X 데이터 (상수항 포함, 포뮬러 연산 적용됨)
    * return : 독립변수와 VIF 값의 데이터 프레임 반환
    """
    
    vif_df = pd.DataFrame()
    vif_df["features"] = data_X.columns
    vif_df["VIF"] = [variance_inflation_factor(data_X.values, i) \
                     for i in range(data_X.shape[1])]
    vif_df = vif_df.sort_values("VIF").reset_index(drop=True)
    
    return vif_df










