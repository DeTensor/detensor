import warnings
# 过滤掉pandas弃用警告
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
import motif
import detensor
import hashlib
import logging
from motif import RemoteVariable as RV
import copy
logger = logging.getLogger("Motif")
logger.setLevel(logging.INFO)
motif.PrivatesLoader.dev = True
motif.PrivatesLoader.myfunc = {
    'get_data': 'http://localhost:7001/get_data'
}


def list_2_df(list_data):
    # 将list数据转换为DataFrame
    # 第一项是列标题
    headers = list_data[0]
    # 其余项是数据
    rows = list_data[1:]
    # 使用列标题和数据创建DataFrame
    df = pd.DataFrame(rows, columns=headers)
    return df

def string_to_63bit_positive_integer(s: str) -> int:
    # 使用 SHA256 哈希算法
    hash_object = hashlib.sha256(s.encode())
    # 获取哈希值的十六进制表示
    hex_hash = hash_object.hexdigest()
    # 将十六进制表示转换为整数
    integer_hash = int(hex_hash, 16)
    # 只保留前63位并将结果转换为正整数
    truncated_hash = integer_hash & 0x7FFFFFFFFFFFFFFF
    return truncated_hash

# 数据预处理
def preprocessing(df:pd.DataFrame) -> pd.DataFrame:
    # 根据前四个字段进行分组聚合,拼接数据 {原始索引：数量}
    grouped = df.groupby(['equipment_name', 'contract_no', 'specifications', 'supplier_code'])
    serious = grouped.apply(lambda x: {row['id']: row['number'] for index, row in x.iterrows()})
    grouped_df = serious.reset_index(name='aggregated')
    grouped_df['key'] = grouped_df['equipment_name'] + '_' + grouped_df['contract_no'] + '_' + grouped_df['specifications'] + '_' + grouped_df['supplier_code']
    grouped_df['hashed_key'] = grouped_df['key'].apply(string_to_63bit_positive_integer)
    return grouped_df

def list_1dto2d(before_list, after_list):
    return np.array([[x, y] for x, y in zip(before_list, after_list)])

def merge_res(a,b,c,d):
    a = list_1dto2d(a, [1] * len(a))
    b = list_1dto2d(b, [2] * len(b))
    c = list_1dto2d(c, [3] * len(c))
    d = list_1dto2d(d, [4] * len(d))
    # 合并数据集
    # 筛选出非空数组
    non_empty_arrays = [arr for arr in [a, b, c, d] if arr.size > 0]

    # 如果存在非空数组，则进行拼接
    if non_empty_arrays:
        match_res = np.concatenate(non_empty_arrays, axis=0)
    else:
        match_res = np.empty((0, 0))  # 如果所有数组都为空，则创建一个空数组
    return pd.DataFrame(match_res, columns=["id", "match_type"])
    
    



def get_key_list(Dict_Data):
    return list(Dict_Data.keys())

def get_value_list(Dict_Data):
    return list(Dict_Data.values())

#根据值删字典中一个元素
def get_pop_dict_1(Dict_Data,value_del):
    del_key = ''
    for key, value in Dict_Data.items():
        if value == value_del:
            Dict_Data.pop(key)
            del_key = key
            break
    return [Dict_Data,del_key]

#根据数组删字典中多个元素
def get_pop_dict_n(Dict_Data, list_num):
    get_keys = []
    for tempnum in list_num[1]:
        for key, value in copy.deepcopy(Dict_Data).items():  
            if value == tempnum:  
                tempnum = -1
                get_keys.append(key) 
                Dict_Data.pop(key)  
                break  
    return [Dict_Data,get_keys]

def get_list():
    return []

def list_append(mylist,v):
    mylist.append(v)
    return mylist

def list_add(list1,list2):
    return list1+list2

def get_key_from_index(key_list,index_list):
    return [key_list[index] for index in index_list]

def delete_key_from_dict(d, key_list):
    return {key: value for key, value in d.items() if key not in key_list}

def match_1_1_mjh(alice_index_num, bob_index_num,C_Site: motif.Site):
    A_Site = alice_index_num.site
    B_Site = bob_index_num.site
    A_value = motif.RemoteCallAt(A_Site)(get_value_list)(alice_index_num)
    B_value = motif.RemoteCallAt(B_Site)(get_value_list)(bob_index_num)
    A_index_list, B_index_list = detensor.psi.match_indices(A_value, B_value, C_Site)
    # 通过index查找对应的key
    A_key = motif.RemoteCallAt(A_Site)(get_key_list)(alice_index_num)
    A_res = motif.RemoteCallAt(A_Site)(get_key_from_index)(A_key, A_index_list)
    alice_index_num = motif.RemoteCallAt(A_Site)(delete_key_from_dict)(alice_index_num, A_res)
    B_key = motif.RemoteCallAt(B_Site)(get_key_list)(bob_index_num)
    B_res = motif.RemoteCallAt(B_Site)(get_key_from_index)(B_key, B_index_list)
    bob_index_num = motif.RemoteCallAt(B_Site)(delete_key_from_dict)(bob_index_num, B_res)
    return alice_index_num, bob_index_num, A_res, B_res

def extend(l1, l2):
    l1.extend(l2)
    return l1

def sum_count_in_dict(d:dict):
    return [sum(d.values())]

def match_1_n_np(alice_index_num:RV, bob_index_num:RV,C_Site: motif.Site):
    A_Site = alice_index_num.site
    B_Site = bob_index_num.site
    index_1_n = motif.RemoteCallAt(A_Site)(get_list)()
    index_1_n.at(A_Site)
    bindex_1_n = motif.RemoteCallAt(B_Site)(get_list)()
    bindex_1_n.at(B_Site)
    A_data_value = motif.RemoteCallAt(A_Site)(get_value_list)(alice_index_num)
    lena = A_data_value.len().at_public().evaluate()
    for i in range(lena):
        max = 1000
        ableflag = False
        for j in range(6):
            B_data_value = motif.RemoteCallAt(B_Site)(get_value_list)(bob_index_num)
            B_combine_list = motif.RemoteCallAt(B_Site)(getlist_n)(max)
            B_combine_list = motif.RemoteCallAt(B_Site)(dealcombine)(B_data_value,B_combine_list,max)
            B_sumlist = motif.RemoteCallAt(B_Site)(combine2sum)(B_combine_list,0)
            len_sumb = B_sumlist.len().at_public().evaluate()
            for k in range(0, len_sumb,3000):
                A_list_num, B_list_num = detensor.psi.compare_2set(A_data_value[i:i+1],B_sumlist[k:k+3000],C_Site)
                a_find_flag = motif.RemoteCallAt(A_Site)(whether_find)(A_list_num)[0].at_public().evaluate()
                if a_find_flag == 0:
                    alice_key = motif.RemoteCallAt(A_Site)(get_pop_dict_1)(alice_index_num, A_data_value[i])
                    alice_index_num = alice_key[0]
                    del_keya = alice_key[1]
                    index_1_n = motif.RemoteCallAt(A_Site)(list_append)(index_1_n,del_keya)
                    b_find_flag = motif.RemoteCallAt(B_Site)(whether_find)(B_list_num)
                    bob_key = motif.RemoteCallAt(B_Site)(get_pop_dict_n)(bob_index_num,B_combine_list[B_sumlist[k+b_find_flag[2].at_public().evaluate()]])
                    bob_index_num = bob_key[0]
                    del_keyb = bob_key[1]
                    bindex_1_n = motif.RemoteCallAt(B_Site)(list_add)(bindex_1_n,del_keyb)
                    ableflag = True
                    break
                elif a_find_flag == -1:
                    ableflag = True

            max = max + max
            if ableflag:
                break

    return  alice_index_num, bob_index_num, index_1_n, bindex_1_n

def getlist_n(n):
    return [[0,[],[],-1] for i in range(n)]

def dealcombine(mylist,mypath,n):
    for i in range(len(mylist)):
        tempnum = mylist[i]
        copypath = copy.deepcopy(mypath)
        for j in range(n):
            temp = j + tempnum
            if temp < n:
                if j == 0 and copypath[temp][3] == -1 :
                    mypath[temp][0] = copypath[j][0] + 1
                    mypath[temp][1] = copypath[j][1] + [tempnum]
                    mypath[temp][2] = copypath[j][2] + [i]
                    mypath[temp][3] = i
                elif copypath[j][3] > -1:
                    if copypath[temp][0] > 2:
                        mypath[temp][0] = 2
                        mypath[temp][1] = [j] + [tempnum]
                        mypath[temp][2] = [copypath[j][3]] + [i]
                    elif copypath[temp][0] == 2 and copypath[temp][2][0] > copypath[j][3]:
                        mypath[temp][0] = 2
                        mypath[temp][1] = [j] + [tempnum]
                        mypath[temp][2] = [copypath[j][3]] + [i]
                    elif copypath[temp][0] < 2:
                        mypath[temp][0] = 2
                        mypath[temp][1] = [j] + [tempnum]
                        mypath[temp][2] = [copypath[j][3]] + [i]
                elif copypath[j][0] >= 1 and copypath[temp][0] <= 1:
                    mypath[temp][0] = copypath[j][0] + 1
                    mypath[temp][1] = copypath[j][1] + [tempnum]
                    mypath[temp][2] = copypath[j][2] + [i]
                elif copypath[j][0] >= 1 and copypath[temp][0] > (copypath[j][0] + 1):
                    mypath[temp][0] = copypath[j][0] + 1
                    mypath[temp][1] = copypath[j][1] + [tempnum]
                    mypath[temp][2] = copypath[j][2] + [i]
                elif copypath[j][0] >= 1 and copypath[temp][0] == (copypath[j][0] + 1):
                    flag = False
                    for k in range(copypath[j][0]):
                        if copypath[j][2][k] > copypath[temp][2][k]:
                            break
                        elif copypath[j][2][k] < copypath[temp][2][k]:
                            flag = True
                            break
                    if flag:
                        mypath[temp][0] = copypath[j][0] + 1
                        mypath[temp][1] = copypath[j][1] + [tempnum]
                        mypath[temp][2] = copypath[j][2] + [i]
    return mypath

def combine2sum(combinelist,base):
    n = len(combinelist)
    sumlist = []
    for i in range(n):
        if combinelist[i][0] > 1:
            sumlist.append(i+base)

    return sumlist

def whether_find(mymatrix):
    flag = 1
    n = len(mymatrix)
    for ii in range(n):
        i = n-1-ii
        for j in range(len(mymatrix[i])):
            if mymatrix[i][j] == 0:
                return [0,i,j]
            elif mymatrix[i][j] == -1:
                flag = -1
    return [flag,-1,-1]

def match_m_n_np(alice_index_num:RV, bob_index_num:RV,C_Site: motif.Site):
    A_Site = alice_index_num.site
    B_Site = bob_index_num.site
    a_sum = motif.RemoteCallAt(A_Site)(sum_count_in_dict)(alice_index_num)
    b_sum = motif.RemoteCallAt(B_Site)(sum_count_in_dict)(bob_index_num)
    a_res, b_res = detensor.psi.compare_2set(a_sum, b_sum, C_Site)
    if a_res.at_public().evaluate() == [[0]]:
        a_remain_index_num = motif.RemoteCallAt(A_Site)(get_list)()
        b_remain_index_num = motif.RemoteCallAt(B_Site)(get_list)()
        a_m_n_matched_keys = motif.RemoteCallAt(A_Site)(get_key_list)(alice_index_num)
        b_m_n_matched_keys = motif.RemoteCallAt(B_Site)(get_key_list)(bob_index_num)
        return a_remain_index_num, b_remain_index_num, a_m_n_matched_keys, b_m_n_matched_keys
    index_m_n = motif.RemoteCallAt(A_Site)(get_list)()
    index_m_n.at(A_Site)
    bindex_m_n = motif.RemoteCallAt(B_Site)(get_list)()
    bindex_m_n.at(B_Site)
    flag1 = True
    while flag1:
        max = 1000
        flag1 = False
        ableflag = False
        A_data_value = motif.RemoteCallAt(A_Site)(get_value_list)(alice_index_num)
        B_data_value = motif.RemoteCallAt(B_Site)(get_value_list)(bob_index_num)
        for i in range(6):
            A_combine_list = motif.RemoteCallAt(A_Site)(getlist_n)(max)
            A_combine_list = motif.RemoteCallAt(A_Site)(dealcombine)(A_data_value,A_combine_list,max)
            B_combine_list = motif.RemoteCallAt(B_Site)(getlist_n)(max)
            B_combine_list = motif.RemoteCallAt(B_Site)(dealcombine)(B_data_value,B_combine_list,max)
            for j in range(0, max, 3000):
                A_sumlist = motif.RemoteCallAt(A_Site)(combine2sum)(A_combine_list[j:j+3000],j)
                B_sumlist = motif.RemoteCallAt(B_Site)(combine2sum)(B_combine_list[j:j+3000],j)
                len_suma = A_sumlist.len().at_public().evaluate()
                len_sumb = B_sumlist.len().at_public().evaluate()
                if len_suma == 0 or len_sumb == 0:
                    continue
                A_list_num, B_list_num = detensor.psi.compare_2set(A_sumlist,B_sumlist,C_Site)
                a_find_flag = motif.RemoteCallAt(A_Site)(whether_find)(A_list_num)
                b_find_flag = motif.RemoteCallAt(B_Site)(whether_find)(B_list_num)
                if a_find_flag[0].at_public().evaluate() == 0:
                    alice_key = motif.RemoteCallAt(A_Site)(get_pop_dict_n)(alice_index_num,A_combine_list[A_sumlist[a_find_flag[1].at_public().evaluate()]])
                    alice_index_num = alice_key[0]
                    del_keya = alice_key[1]
                    index_m_n.extend(del_keya)
                    bob_key = motif.RemoteCallAt(B_Site)(get_pop_dict_n)(bob_index_num,B_combine_list[B_sumlist[b_find_flag[2].at_public().evaluate()]])
                    bob_index_num = bob_key[0]
                    del_keyb = bob_key[1]
                    bindex_m_n.extend(del_keyb)
                    flag1 = True
                    ableflag = True
                    break
                if ableflag:
                    break
            max = max + max
            if ableflag:
                break
    return alice_index_num, bob_index_num, index_m_n, bindex_m_n

def match_main_np(alice_psi:motif.RemoteVariable, bob_psi:motif.RemoteVariable, C_Site: motif.Site):
    # 进行匹配
    A_Site = alice_psi.site
    B_Site = bob_psi.site
    index_1_1_A_all = motif.RemoteCallAt(A_Site)(get_list)()
    index_1_n_A_all = motif.RemoteCallAt(A_Site)(get_list)()
    index_m_1_A_all = motif.RemoteCallAt(A_Site)(get_list)()
    index_m_n_A_all = motif.RemoteCallAt(A_Site)(get_list)()
    index_1_1_B_all = motif.RemoteCallAt(B_Site)(get_list)()
    index_1_n_B_all = motif.RemoteCallAt(B_Site)(get_list)()
    index_m_1_B_all = motif.RemoteCallAt(B_Site)(get_list)()
    index_m_n_B_all = motif.RemoteCallAt(B_Site)(get_list)()
    print(" Start matching  ")
    key_colums_len = alice_psi.shape[0].at_public().evaluate()
    for index in range(key_colums_len):  #双方数据都已经根据key排序过
        alice_index_num = alice_psi.iloc[index]['aggregated']
        bob_index_num = bob_psi.iloc[index]['aggregated']
        alice_index_num, bob_index_num, index_1_1_A, index_1_1_B = match_1_1_mjh(alice_index_num, bob_index_num, C_Site)
        index_1_1_A_all = motif.RemoteCallAt(A_Site)(extend)(index_1_1_A_all, index_1_1_A)
        index_1_1_B_all = motif.RemoteCallAt(B_Site)(extend)(index_1_1_B_all, index_1_1_B)
        alice_index_num, bob_index_num, index_1_n_A, index_1_n_B = match_1_n_np(alice_index_num, bob_index_num, C_Site)
        index_1_n_A_all = motif.RemoteCallAt(A_Site)(extend)(index_1_n_A_all, index_1_n_A)
        index_1_n_B_all = motif.RemoteCallAt(B_Site)(extend)(index_1_n_B_all, index_1_n_B)
        bob_index_num, alice_index_num, index_m_1_B, index_m_1_A = match_1_n_np(bob_index_num, alice_index_num, C_Site)
        index_m_1_A_all = motif.RemoteCallAt(A_Site)(extend)(index_m_1_A_all, index_m_1_A)
        index_m_1_B_all = motif.RemoteCallAt(B_Site)(extend)(index_m_1_B_all, index_m_1_B)
        alice_index_num, bob_index_num, index_m_n_A, index_m_n_B = match_m_n_np(alice_index_num, bob_index_num, C_Site)
        index_m_n_A_all = motif.RemoteCallAt(A_Site)(extend)(index_m_n_A_all, index_m_n_A)
        index_m_n_B_all = motif.RemoteCallAt(B_Site)(extend)(index_m_n_B_all, index_m_n_B)
        logger.info(f'match {index}/{key_colums_len}')

    res_A = motif.RemoteCallAt(A_Site)(merge_res)(index_1_1_A_all,index_1_n_A_all,index_m_1_A_all,index_m_n_A_all)
    res_B = motif.RemoteCallAt(B_Site)(merge_res)(index_1_1_B_all,index_1_n_B_all,index_m_1_B_all,index_m_n_B_all)
    return res_A, res_B

def filter_df(df, common_index):
    df = df[df.index.isin(common_index)]
    return df.sort_values('hashed_key')

def final_res(raw_df, index_df):
    return pd.merge(raw_df, index_df, how='left', on="id").sort_values(["id"])

def df_save_csv(df:pd.DataFrame, filename):
    output_dir = '/usr/src/demo/python-extension/output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    full_path = os.path.join(output_dir, filename)
    return df.to_csv(full_path, index=False)

def main(a_file, b_file):
    # 参与方声明
    A = motif.Site(1) # Invoice
    B = motif.Site(2) # Receipt
    C = motif.Site(3) # 辅助节点
    # 私有接口绑定
    get_a_data = motif.private_interface('get_data', A)
    get_b_data = motif.private_interface('get_data', B) # invoice.csv

    # 调用私有接口获取私有数据
    a_data = get_a_data(name=a_file)
    b_data = get_b_data(name=b_file)

    # 将数据在各自节点转换成pd.DataFrame
    a_df = motif.RemoteCallAt(A)(list_2_df)(a_data)
    b_df = motif.RemoteCallAt(B)(list_2_df)(b_data)
    # 数据预处理 在各自节点执行
    # 根据'equipment_name', 'contract_no', 'specifications', 'supplier_code'聚合 新列aggregated为{原id:number}
    a_grouped_df = motif.RemoteCallAt(A)(preprocessing)(a_df)
    b_grouped_df = motif.RemoteCallAt(B)(preprocessing)(b_df)

    # 将2方的字段编码进行隐私求交，
    # 存储各列编码的交集
    a_common_index ,b_common_index = detensor.psi.psi(a_grouped_df['hashed_key'],b_grouped_df['hashed_key'], C)

    # 根据交集，筛选出交集的行并进行匹配
    # 处理成输入match函数的数据,例如 第一列是编码，第二列是索引和数值的集合 {index:number}
    a_match_input = motif.RemoteCallAt(A)(filter_df)(a_grouped_df, a_common_index)
    b_match_input = motif.RemoteCallAt(B)(filter_df)(b_grouped_df, b_common_index)
    
    # 开始匹配对应关系，返回对应index和type
    index_A,index_B = match_main_np(a_match_input, b_match_input, C)
    # index_all_df与receipt_df进行合并，在最右侧加入 type 列
    res_A = motif.RemoteCallAt(A)(final_res)(a_df, index_A)
    res_B = motif.RemoteCallAt(B)(final_res)(b_df, index_B)
    
    # 结果保存为csv文件    
    motif.RemoteCallAt(A)(df_save_csv)(res_A,"resA.csv").evaluate()
    motif.RemoteCallAt(B)(df_save_csv)(res_B,"resB.csv").evaluate()
    logger.info(f'save result to csv file finish.')

if __name__ == "__main__":
    # 默认输入的数据
    receipt_file = "receipt10.csv"
    invoice_file = "invoice10.csv"

    # 如果有命令行参数则覆盖
    if len(sys.argv) > 2:
        receipt_file = sys.argv[1]
        invoice_file = sys.argv[2]

    import time
    t1 = time.time()
    main(receipt_file, invoice_file)
    t2 = time.time()
    print(f'TIME COST: {t2-t1} s')
    logger.info(f'TIME COST: {t2-t1} s')