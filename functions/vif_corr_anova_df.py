"""
<VIF, corr, ANOVA 데이터 프레임을 병합하는 함수>

모델 결과 객체와 X 데이터는 함께 사용한 것이어야 함

calc_vif, calc_corr, calc_anova_lm 함수 호출하여 측정

VIF, corr, ANOVA 데이터 프레임을 병합한 후 반환

상관관계 행렬 반환 :
    히트맵에서 사용
"""

def vif_corr_anova_df(data_X, result, thr) : 
    
    """
    * data_X : X 데이터 프레임 (result에 사용한 X 데이터)
    * result : 모델의 결과 객체 (data_X를 사용한 모델의 객체)
    * thr : calc_corr 함수의 파라미터
        - 상관관계 갯수 측정의 기준
    * return :
        - 상관관계 행렬
        - vif, corr, anova 데이터를 병합한 데이터 프레임
    """

    vif_df = calc_vif(data_X)
    corr_matrix, corr_df = calc_corr(data_X, thr=thr)
    anova_df = calc_anova_lm(result)
    vif_corr_df = pd.merge(vif_df, corr_df, 
                              left_on="features", right_on="features", how="outer")
    vif_corr_anova = pd.merge(vif_corr_df, anova_df, 
                                left_on="features", right_on="features", how="outer")
    
    return corr_matrix, vif_corr_anova










