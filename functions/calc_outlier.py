"""
<폭스추천 아웃라이어를 측정하여 아웃라이어를 제거한 데이터 프레임을 반환하는 함수>

모수 추정한 모델에서 영향도 행렬 객체 사용
영향도 행렬 객체에서 cooks_distnace 값을 반환하여 폭스 추천값을 계산

cooks_distance에서 폭스 추천값보다 큰 것을 아웃라이어 인덱스로 반환

X, y 데이터 프레임에서 폭스 추천 아웃라이어를 제거 후 병합 :
    단, 모델의 모수 추정에 사용한 X 데이터를 입력해주어야 함
    modeling_dmatirx 모델에서 사용한 것 : 상수항 포함 dmatrix X 데이터
    modeling_non_const 모델에서 사용한 것 : 상수항 미포함 X 데이터
"""

def calc_outlier(result, data_X, data_y) : 
    
    """
    * result : 모델링의 결과 객체
    * data_X : 모델링에 사용한 X 데이터 프레임 
    * data_y : 모델링에 사용한 y 데이터 프레임
    * return : 아웃라이어 인덱스, 아웃라이어 제거 한 인덱스, 아웃라이어 제거한 데이터 프레임 반환
    """
    
    ## 영향도 행렬로부터 쿡스디스턴스 값을 반환받아 폭스 추천값을 계산
    pred = result.predict(data_X)
    influence = result.get_influence()
    cooks_d2, pvalue = influence.cooks_distance
    K = influence.k_vars
    fox = 4 / (len(data_y) - K - 1)
    
    ## 폭스 추천 아웃라이어 + 종속값이 50인 데이터를 제거
    fox_idx = np.where(cooks_d2 > fox)
    fox_50_idx = np.hstack([fox_idx[0], np.where(data_y == 50)[0]])
    idx_non_outlier = list(set(range(len(data_X))).difference(fox_50_idx))
    ## X, y 데이터 프레임에서 아웃라이어를 제거한 후 병합
    df_non_outlier = pd.concat([data_X.iloc[idx_non_outlier], 
                                data_y.iloc[idx_non_outlier]], axis=1)
    
    return fox_50_idx, idx_non_outlier, df_non_outlier










