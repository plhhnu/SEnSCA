import pandas as pd
import numpy as np
from time import time
import os
itime = time()
folders_to_create = [
    r"melanoma\specific_expression",
    r"melanoma\expression_product",
    r"melanoma\cell_expression",
    r"melanoma\Three",
    r"melanoma\Three\TOP",
    r"melanoma"]
for folder in folders_to_create:
    os.makedirs(folder, exist_ok=True)
cancer = r'melanoma'
cell_type = 6
spe = pd.read_csv("The_specific_expression_data.csv", header=None, index_col=None).to_numpy()  # melanoma
pro = pd.read_csv("The_expression_product_data.csv", header=None, index_col=None).to_numpy()  # melanoma
cell = pd.read_csv("The_cell_expression_data.csv", header=None, index_col=None).to_numpy()  # melanoma
LRI = pd.read_csv("LRI.csv", header=None, index_col=None).to_numpy()

#  The specific expression
for w in range(0, LRI.shape[0]):
    zhibiao1 = 0
    zhibiao2 = 0
    flag = False
    for x in range(0, spe.shape[0]):
        if LRI[w][0] == spe[x][0]:
            b11 = spe[x]
            zhibiao1 = 1
        if LRI[w][1] == spe[x][0]:
            b12 = spe[x]
            zhibiao2 = 1
        if zhibiao1 == 1 and zhibiao2 == 1:
            flag = True
            break
    if not flag:
        continue
    b11_l = b11[0]
    b11_r = b12[0]
    LRI_com = "{}_{}".format(b11_l, b11_r)
    b11 = np.delete(b11, 0)
    b12 = np.delete(b12, 0)
    for i in range(0, cell_type + 1):
        for j in range(0, cell_type + 1):
            specific_ = b11[i] * b12[j]
            with open(cancer + "\specific_expression\\" + str(i) + str(j) + ".csv", mode="a") as f:
                f.write("{},{}\n".format(LRI_com, specific_))
            f.close()
for i in range(0, cell_type + 1):
    for j in range(0, cell_type + 1):
        a1 = pd.read_csv(cancer + "\specific_expression\\" + str(i) + str(j) + ".csv", header=None, index_col=None).to_numpy()  # Obtain the result
        SUM = sum(a1[:, 1])
        with open(cancer + "\\specific_result.csv", mode="a") as f:
            f.write(str(i))
            f.write(str(j))
            f.write('____')
            f.write(str(SUM))
            f.write('\n')
        f.close()

#  The expression product
for w in range(0, LRI.shape[0]):
    zhibiao1 = 0
    zhibiao2 = 0
    flag = False
    for x in range(0, pro.shape[0]):
        if LRI[w][0] == pro[x][0]:
            b11 = pro[x]
            zhibiao1 = 1
        if LRI[w][1] == pro[x][0]:
            b12 = pro[x]
            zhibiao2 = 1
        if zhibiao1 == 1 and zhibiao2 == 1:
            flag = True
            break
    if not flag:
        continue
    b11_l = b11[0]
    b11_r = b12[0]
    LRI_com = "{}_{}".format(b11_l, b11_r)
    b11 = np.delete(b11, 0)
    b12 = np.delete(b12, 0)
    for i in range(0, cell_type + 1):
        for j in range(0, cell_type + 1):
            product_ = b11[i] * b12[j]
            with open(cancer + "\expression_product\\" + str(i) + str(j) + ".csv", mode="a") as f:
                f.write("{},{}\n".format(LRI_com, product_))
            f.close()
for i in range(0, cell_type + 1):
    for j in range(0, cell_type + 1):
        a1 = pd.read_csv(cancer + "\expression_product\\" + str(i) + str(j) + ".csv", header=None, index_col=None).to_numpy()  # Obtain the result
        SUM = sum(a1[:, 1])
        with open(cancer + "\\product_result.csv", mode="a") as f:
            f.write(str(i))
            f.write(str(j))
            f.write('____')
            f.write(str(SUM))
            f.write('\n')
        f.close()

# The cell expression
for w in range(0, LRI.shape[0]):
    zhibiao1 = 0
    zhibiao2 = 0
    flag = False
    for x in range(0, cell.shape[0]):
        if LRI[w][0] == cell[x][0]:
            b11 = cell[x]
            zhibiao1 = 1
        if LRI[w][1] == cell[x][0]:
            b12 = cell[x]
            zhibiao2 = 1
        if zhibiao1 == 1 and zhibiao2 == 1:
            flag = True
            break
    if not flag:
        continue
    b11_l = b11[0]
    b11_r = b12[0]
    LRI_com = "{}_{}".format(b11_l, b11_r)
    b11 = np.delete(b11, 0)
    b12 = np.delete(b12, 0)
    for i in range(0, cell_type + 1):
        for j in range(0, cell_type + 1):
            cell_ = b11[i] * b12[j]
            with open(cancer + "\cell_expression\\" + str(i) + str(j) + ".csv", mode="a") as f:
                f.write("{},{}\n".format(LRI_com, cell_))
            f.close()
for i in range(0, cell_type + 1):
    for j in range(0, cell_type + 1):
        a1 = pd.read_csv(cancer + "\cell_expression\\" + str(i) + str(j) + ".csv", header=None, index_col=None).to_numpy()  # Obtain the result
        SUM = sum(a1[:, 1])
        with open(cancer + "\\cell_result.csv", mode="a") as f:
            f.write(str(i))
            f.write(str(j))
            f.write('____')
            f.write(str(SUM))
            f.write('\n')
        f.close()

explain = "The xxx_result.csv file indicates that 0 represents the Melanoma cancer cells,\n1 represents the T cells," \
          "\n2 represents the B cells,\n3 represents the Macrophages,\n4 represents the Endothelial cells," \
          "\n5 represents the CAFs,\n6 represents the NK cells.\nFor example: 12_xxx represents the communication " \
          "between T cells and B cells, and xxx is the calculated communication strength. "
print('--------------------------------------------------------------')
print(explain)

sum_data = 0
data = pd.read_csv(cancer + "\\specific_result.csv", header=None,index_col=None).to_numpy()
data2 = []
for j in range(len(data)):
    data11 = data[j][0]
    data1 = float(data11[6:])
    a1 = data11[:2]
    if a1[0] == a1[1]:
        data1 = 0.0
    data2.append(data1)
_range = np.max(data2) - np.min(data2)
aaa = (data2 - np.min(data2)) / _range
sum_data = pd.DataFrame(aaa)
totol = sum_data.values
result = totol.reshape((cell_type + 1, cell_type + 1))
result1 = pd.DataFrame(result)
result1.to_csv(cancer + "\\result1.csv", header=None, index=None)

sum_data = 0
data = pd.read_csv(cancer + "\\product_result.csv", header=None,index_col=None).to_numpy()
data2 = []
for j in range(len(data)):
    data11 = data[j][0]
    data1 = float(data11[6:])
    a1 = data11[:2]
    if a1[0] == a1[1]:
        data1 = 0.0
    data2.append(data1)
_range = np.max(data2) - np.min(data2)
aaa = (data2 - np.min(data2)) / _range
sum_data = pd.DataFrame(aaa)
totol = sum_data.values
result = totol.reshape((cell_type + 1, cell_type + 1))
result2 = pd.DataFrame(result)
result2.to_csv(cancer + "\\result2.csv", header=None, index=None)

sum_data = 0
data = pd.read_csv(cancer + "\\cell_result.csv", header=None,index_col=None).to_numpy()
data2 = []
for j in range(len(data)):
    data11 = data[j][0]
    data1 = float(data11[6:])
    a1 = data11[:2]
    if a1[0] == a1[1]:
        data1 = 0.0
    data2.append(data1)
_range = np.max(data2) - np.min(data2)
aaa = (data2 - np.min(data2)) / _range
sum_data = pd.DataFrame(aaa)
totol = sum_data.values
result = totol.reshape((cell_type + 1, cell_type + 1))
result3 = pd.DataFrame(result)
result3.to_csv(cancer + "\\result3.csv", header=None, index=None)

# The three-point estimation method
result_max = np.maximum(result1, result2)
result_med = np.median([result1, result2, result3], axis=0)
result_min = np.minimum(result1, result2)
result_matrix = np.maximum(result_max, result3) + result_med * 4 + np.minimum(result_min, result3)
result_matrix /= 6
result_matrix = pd.DataFrame(result_matrix)
result_matrix.to_csv(cancer + "\\result_matrix.csv", header=None, index=None)

