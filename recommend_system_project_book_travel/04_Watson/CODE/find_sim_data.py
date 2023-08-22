import numpy as np


        
def find_sim_data(df, sorted_ind, code):
    code = df[df['code'] == code]
    code_index = code.index.values
    indexes = sorted_ind[code_index, :]
    #print(code_index)
    #print(indexes)
 
    #책일경우
    if code_index < 200:
        similar_indexes = []
        for idx in indexes[0]:
            # 여행데이터에서 가장 비슷한 것 순으로 정렬
            if idx >= 200:
                similar_indexes.append(idx)
        #print(similar_indexes)
  
    #여행일경우
    else:
        similar_indexes = []
        for idx in indexes[0]:
            # 책데이터에서 가장 비슷한 것 순으로 정렬
            if idx < 200:
                similar_indexes.append(idx)
        #print(similar_indexes)
        
    similar_indexes = np.array(similar_indexes).reshape(-1)

    return df.iloc[similar_indexes]
