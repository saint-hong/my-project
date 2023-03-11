"""
<PCA를 적용한 모델을 반환하는 함수>

데이터에 PCA를 적용하여 차원축소 후 OLS 모델로 모수추정

차원축소 후 다시 원래 차원으로 변환 된 행렬, 차원 축소 데이터, 차원 축소 모델 객체 반환

사용한 패키지 : 
    from sklearn.decomposition import PCA
    import pandas as pd
"""

def pca_modeling(n, df_X, formula) :

    """
    * n : 차원축소의 주성분값
    * df_X : 상수항 미포함, X,y 병합 데이터
    * formula : str의 formula 식
    * return : 역변환 된 행렬, 차원축소 된 행렬, 차원축소 데이터를 적용한 모델 결과 객체
    """

    ## PCA 객체 생성
    pca = PCA(n_components=n)
    pca_X = pca.fit_transform(df_X)
    ## 역변환 된 행렬 : 차원축소 -> 원래 차원으로 복구
    inverse_X = pca.inverse_transform(pca_X)
    
    ## 차원축소 된 X 데이터
    pca_X_df = pd.DataFrame(pca_X, columns=["comp_{}".format(i) for i in range(1, n+1)])
    df_y = df_4["MEDV"].reset_index(drop=True)
    pca_df = pd.concat([pca_X_df, df_y], axis=1)

    ## 차원축소를 적용한 데이터로 모델링
    pca_model, pca_result = modeling_non_const("MEDV ~ " + formula, pca_df)

    return inverse_X, pca_df, pca_result










