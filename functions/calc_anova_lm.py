"""
<ANOVA 데이터 프레임을 반환하는 함수>
    
분산분석 ANOVA 값과 유의확률을 데이터 프레임으로 반환

mdeling_non_const 함수로 생성한 모델을 사용해야함 :
    상수항 미포함, X, y 병합 데이터 사용한 모델

사용한 패키지 : 
    sm.stats.anova_lm()
"""

def calc_anova_lm(result) : 
    
    """
    * result : modeling_non_const 함수로 만든 모델의 결과 객체
        - 상수항 미포함, X, y 병합 데이터 사용
    * return : ANOVA 값과 유의확률을 데이터 프레임으로 반환
    """

    anova = sm.stats.anova_lm(result, typ=2)[["F", "PR(>F)"]]
    anova["PR(>F)"] = anova["PR(>F)"].map(lambda x : round(x, 6))
    ## 인덱스 이름 Residual -> Intercept로 변환
    anova.index = [[i if i != "Residual" else "Intercept"][0] for i in anova.index]
    anova_df = pd.DataFrame(anova).reset_index().rename(columns={"index":"features", "F":"ANOVA_F"})
    
    return anova_df










