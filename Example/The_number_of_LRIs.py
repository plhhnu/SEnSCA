import pandas as pd
import numpy as np
import os
cancer = r'melanoma'
x_ytick = ['Melanoma cancer cells', 'T cells', 'B cells', 'Macrophages', 'Endothelial cells', 'CAFs ', 'NK cells'] # Melanoma->0...NK->6
cell_type = 6  
sum_data = 0
for i in range(0, cell_type + 1):
    for j in range(0, cell_type + 1):
        list1 = pd.read_csv(cancer + "\specific_expression\\" + str(i) + str(j) + ".csv", header=None, index_col=None).to_numpy()  # 得分
        _range1 = np.max(list1[:, 1]) - np.min(list1[:, 1])
        aaa1 = (list1[:, 1] - np.min(list1[:, 1])) / _range1
        count = np.sum(aaa1 > 0.05)
        with open(cancer + "\\spe_LRi_num.csv",mode="a") as f:
            f.write(str(i))
            f.write(str(j))
            f.write('____')
            f.write(str(count))
            f.write('\n')
        f.close()
for i in range(0, cell_type + 1):
    for j in range(0, cell_type + 1):
        list2 = pd.read_csv(cancer + "\expression_product\\" + str(i) + str(j) + ".csv", header=None, index_col=None).to_numpy()  # 得分
        _range2 = np.max(list2[:, 1]) - np.min(list2[:, 1])
        aaa2 = (list2[:, 1] - np.min(list2[:, 1])) / _range2
        count = np.sum(aaa2 > 0.05)
        with open(cancer + "\\pro_LRi_num.csv",mode="a") as f:
            f.write(str(i))
            f.write(str(j))
            f.write('____')
            f.write(str(count))
            f.write('\n')
        f.close()
for i in range(0, cell_type + 1):
    for j in range(0, cell_type + 1):
        list3 = pd.read_csv(cancer + "\cell_expression\\" + str(i) + str(j) + ".csv", header=None, index_col=None).to_numpy()  # 得分
        _range3 = np.max(list3[:, 1]) - np.min(list3[:, 1])
        aaa3 = (list3[:, 1] - np.min(list3[:, 1])) / _range3
        count = np.sum(aaa3 > 0.05)
        with open(cancer + "\\cell_LRi_num.csv",mode="a") as f:
            f.write(str(i))
            f.write(str(j))
            f.write('____')
            f.write(str(count))
            f.write('\n')
        f.close()

for i in range(0, cell_type + 1):
    for j in range(0, cell_type + 1):
        
        list1 = pd.read_csv(cancer + "\specific_expression\\" + str(i) + str(j) + ".csv", header=None, index_col=None).to_numpy()
        _range1 = np.max(list1[:, 1]) - np.min(list1[:, 1])
        aaa1 = (list1[:, 1] - np.min(list1[:, 1])) / _range1

        list2 = pd.read_csv(cancer + "\expression_product\\" + str(i) + str(j) + ".csv", header=None, index_col=None).to_numpy()
        _range2 = np.max(list2[:, 1]) - np.min(list2[:, 1])
        aaa2 = (list2[:, 1] - np.min(list2[:, 1])) / _range2

        list3 = pd.read_csv(cancer + "\cell_expression\\" + str(i) + str(j) + ".csv", header=None, index_col=None).to_numpy()
        _range3 = np.max(list3[:, 1]) - np.min(list3[:, 1])
        aaa3 = (list3[:, 1] - np.min(list3[:, 1])) / _range3
        
        aaa = (aaa1 + aaa2 + aaa3) / 3
        gene = list1[:, 0]
        gene_val = np.column_stack((gene, aaa))
        df = pd.DataFrame(gene_val)
        output_path = cancer + "\Three"
        output_name = '{}{}.csv'
        output_file = os.path.join(output_path, output_name.format(i, j))
        df.to_csv(output_file, index=False, header=False)
        count = np.sum(aaa > 0.05)
        with open(cancer + "\\Three_LRi_num.csv", mode="a") as f:
            f.write(str(i))
            f.write(str(j))
            f.write('____')
            f.write(str(count))
            f.write('\n')
        f.close()
data = pd.read_csv(cancer + "\\Three_LRi_num.csv", header=None, index_col=None).to_numpy()
data2 = []
for j in range(len(data)):
    data11 = data[j][0]
    data1 = float(data11[6:])
    a1 = data11[:2]
    if a1[0] == a1[1]:
        data1 = 0.0
    data2.append(data1)
sum_data = pd.DataFrame(data2)

totol = sum_data.values
result = totol.reshape((cell_type + 1, cell_type + 1))
result = pd.DataFrame(result)
result.to_csv(cancer + "\\result.csv", header=None, index=None)
