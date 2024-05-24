import numpy as np
import time
import motif
from motif.RemoteVariable import RemoteVariable as RV
from detensor.psi import psi

# 生成随机矩阵，形状为shape
def generate_random_set(size):
    return np.random.randint(low=-1000, high=1000, size=size)

motif.PrivatesLoader.dev = True
motif.PrivatesLoader.mydata = {
    2:{'B_data': generate_random_set(2000)},
    3:{'C_data': generate_random_set(2000)}
}

def time_test_for_psi_internal(A_data: RV, B_data: RV, C_Site: motif.Site):
    A_res, B_res = psi(A_data, B_data, C_Site)
    return A_res , B_res

def subtract_lists_matrix(a, b):
    # 获取列表 a 和列表 b 的长度
    rows = len(a)
    cols = len(b)
    
    # 构建矩阵，矩阵的每个元素都是列表 a 的第 i 项减去列表 b 的第 j 项的结果
    matrix = [[a[i] - b[j] for j in range(cols)] for i in range(rows)]
    
    return np.array(matrix)

def get_zero_indices(tmp):
    zero_indices = np.argwhere(np.isclose(tmp, 0))
    return np.unique(zero_indices[:, 0]), np.unique(zero_indices[:, 1])

def judge(A1,B1,A2,B2):
    if len(A1)!=len(A2) or len(B1)!=len(B2):
        print("len not equal")
        print(A1)
        print(A2)
        print(B1)
        print(B2)
        return
    for i in range(len(A1)):
        if A1[i]!=A2[i]:
            print("A1A2 not equal")
            return
    for i in range(len(B1)):
        if B1[i]!=B2[i]:
            print("B1B2 not equal")
            return
    print("correct")
    
def transform_matrix(matrix):
    # 将大于0的元素变成1，小于0的元素变成-1，等于0的元素变成0
    transformed_matrix = np.where(matrix > 0, 1, np.where(matrix < 0, -1, 0))
    return transformed_matrix

def main():
    A_Site = motif.Site(1)
    B_Site = motif.Site(2)
    C_Site = motif.Site(3)
    D_Site = motif.Site(4)
    B_data = motif.private_data("B_data", B_Site)
    C_data = motif.private_data("C_data", C_Site)
    
    # 测试psi
    start = time.time()
    A_res1 ,B_res1 = time_test_for_psi_internal(B_data, C_data, D_Site)
    A_res1 = A_res1.at(A_Site)
    B_res1 = B_res1.at(A_Site)
    print("time=", time.time()-start)

    set1_element = B_data.at(A_Site)
    set2_element = C_data.at(A_Site)
    ans_matrix = motif.RemoteCallAt(A_Site)(subtract_lists_matrix)(set1_element, set2_element)
    res = motif.RemoteCallAt(A_Site)(get_zero_indices)(ans_matrix)
    A_res2 = res[0]
    B_res2 = res[1]
    motif.RemoteCallAt(A_Site)(judge)(A_res1,B_res1,A_res2,B_res2).evaluate()
    
if __name__ == "__main__":
    main()