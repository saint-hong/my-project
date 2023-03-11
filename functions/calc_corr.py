"""
<상관관계 데이터프레임과 행렬을 반환하는 함수>
   
dmatrix_X_df로 변환된 X 데이터 사용

독립변수별로 상관관계 평균값과 thr보다 큰 갯수를  데이터 프레임으로 반환
    
big_corr_count 컬럼 :
    상관관계값이 thr 보다 큰 것의 갯수 thr의 디폴트 0.5

히트맵에서 사요할 수 있는 상관관계 행렬 반환
"""

def calc_corr(data_X, thr=0.5) : 
    
    """    
    * data_X : dmatrix에서 반환 된 X 데이터 (상수항 포함, 포뮬러 연산 적용됨)
    * thr : 상관관계 크기의 기준값
        - thr 보다 큰 값의 갯수를 계산하기 위함
    * return : 
        - 상관관계 행렬 : 히트맵으로 시각화 가능
        - 데이터 프레임 
    """
    
    corr_matrix = data_X.corr()
    
    big_corr_count = [0] * len(corr_matrix)
    for i in range(len(corr_matrix)) : 
        big_corr_count[i] = corr_matrix.iloc[i].apply(lambda x : 
                                             1 if abs(x) > thr else 0).sum()
    
    mean_count_corr = corr_matrix.mean(axis=1).reset_index()  
    mean_count_corr.columns = ["features", "corr_mean"]
    mean_count_corr["big_corr_count"] = big_corr_count
    
    return corr_matrix, mean_count_corr









