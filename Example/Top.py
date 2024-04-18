import pandas as pd
import numpy as np
cancer = r'melanoma'
cell_type = 6
top = 3
X_tick = pd.read_csv("X_ticklabels.csv", header=None, index_col=None)
cell_cell_num = cell_type * 2
data2 = []
top_three = pd.DataFrame()
for i in range(0, cell_type + 1):
    for j in range(0, cell_type + 1):
        if (i == 0 and j != 0) or (i != 0 and j == 0):
            list1 = pd.read_csv(cancer + "\specific_expression\\" + str(i) + str(j) + ".csv",header=None, index_col=None)
            list2 = pd.read_csv(cancer + "\expression_product\\" + str(i) + str(j) + ".csv",header=None, index_col=None)
            list3 = pd.read_csv(cancer + "\cell_expression\\" + str(i) + str(j) + ".csv",header=None, index_col=None)
            _range = np.max(list1[1]) - np.min(list1[1])
            list11 = (list1[1] - np.min(list1[1])) / _range
            _range = np.max(list2[1]) - np.min(list2[1])
            list22 = (list2[1] - np.min(list2[1])) / _range
            _range = np.max(list3[1]) - np.min(list3[1])
            list33 = (list3[1] - np.min(list3[1])) / _range

            list = (list11 + list22 + list33)/3
            a = [list1[0], list]
            train = pd.concat(a, axis=1, ignore_index=True)
            df1 = train.sort_values(by=1, ascending=False, ignore_index=True)
            three = df1[0].head(top)
            top_three = pd.concat([top_three, three], axis=0)
            top_three = top_three.drop_duplicates()
            top_three = top_three.sort_values(by=top_three.columns[0])
            top_three = top_three.reset_index(drop=True)
            df1.to_csv(cancer + "\Three\TOP\\" + str(i) + str(j) + ".csv", index = False,header = False)

for i in range(0, cell_type + 1):
    for j in range(0, cell_type + 1):
        if (i == 0 and j != 0) or (i != 0 and j == 0):
            list = pd.read_csv(cancer + "\Three\TOP\\" + str(i) + str(j) + ".csv",header=None, index_col=None)
            for w in range(0, top_three.shape[0]):
                for x in range(0, list.shape[0]):
                    if top_three[0][w] == list[0][x]:
                        b1 = list[1][x]
                        b1 = b1.astype(float)
                        data2.append(b1)
                        break
sum_data = pd.DataFrame(data2)
totol = sum_data.values
result = totol.reshape((cell_cell_num, top_three.shape[0]))
result = result.T
result = pd.DataFrame(result)

ytick = top_three[0].tolist()
xtick = X_tick[0].tolist()
df = result.copy()
df.index = ytick
df.columns = xtick
df.index.name = 'cell_type'
df.to_csv(cancer + "\Three\TOP\\Top_data.csv", index=True)

