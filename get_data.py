def get_trajectory():
    import pandas as pd
    import numpy as np
    file = '3.csv'
    t0 = 1118936700000
    csv_data = pd.read_csv(file)

    needed_ID = [378, 451, 459]

    needed_tra = []
    for i in range(len(needed_ID)):
        v2 = csv_data[(csv_data['Vehicle_ID'] == needed_ID[i])]
        v2_location = v2[['Global_Time', 'Local_Y', 'v_Vel', 'Vehicle_ID']]

        # 产生测试数据

        data1_spa_1 = np.array(v2_location['Global_Time'])
        data1_spa_2 = np.array(v2_location['Local_Y'])
        data1_spa_5 = np.array(v2_location['v_Vel'])
        t = (np.array(data1_spa_1[:]) - t0) / 1000

        need_t = t[:len(data1_spa_2[data1_spa_2 < 500])]
        need_y = data1_spa_2[data1_spa_2 < 500]
        need_v = data1_spa_5[:len(data1_spa_2[data1_spa_2 < 500])]
        needed_tra.append([need_t, need_y, need_v, needed_ID[i]])


    delet_arr = [i for i in range(len(needed_ID))]
    for de in delet_arr:
        numeber_of_delet = 0
        for i in range(len(needed_tra[de][1])):
            if needed_tra[de][2][i] > 5:
                break
            else:
                numeber_of_delet += 1
        needed_tra[de][0] = needed_tra[de][0][numeber_of_delet:]
        needed_tra[de][1] = needed_tra[de][1][numeber_of_delet:]

    return needed_tra
