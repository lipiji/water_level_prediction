from datetime import datetime
import csv
import numpy as np
import json
import pickle

'''
date_list = []
f_climate_all = {}

day_feature = []
with open('pylb_climate_hydro_daily.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        fs = row[0].split(",")
        if fs[0] == "Year":
            continue
            
            
        f_climate = [float(i) for i in fs[11:19]]

        date = fs[1] + " " + fs[2] + " " + fs[0]
        date_obj = datetime.strptime(date, '%m %d %Y')
        date_str = date_obj.strftime("%Y%m%d")
        if date_str not in date_list:
            if len(day_feature) > 0:
                day_feature_ary = np.array(day_feature)
                f_climate_all[date_list[-1]] = day_feature_ary.mean(axis=0)
            date_list.append(date_str)
            day_feature = []
        day_feature.append(f_climate)
        
    if len(day_feature) > 0:
        day_feature_ary = np.array(day_feature)
        f_climate_all[date_str] = day_feature_ary.mean(axis=0)


    print(len(date_list), len(f_climate_all))

    with open('f_climate.pk', 'wb') as f:
        #json.dump([date_list, f_climate_all], f, ensure_ascii=False, indent=4)
        pickle.dump([date_list, f_climate_all], f, pickle.HIGHEST_PROTOCOL)

f_quantity_all = {}
f_hydro_all = {}

with open('pylb_hydro_daily.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        fs = row[0].split(",")
        if fs[0] == "Year":
            continue
            
            
        f_quantity = [float(i) if i else 0 for i in fs[3:8]]
        f_hydro = [float(i) if i else 0 for i in fs[9:14]]

        date = fs[1] + " " + fs[2] + " " + fs[0]
        date_obj = datetime.strptime(date, '%m %d %Y')
        date_str = date_obj.strftime("%Y%m%d")
        f_quantity_all[date_str] = f_quantity
        f_hydro_all[date_str] = f_hydro


    print(len(f_quantity_all), len(f_hydro_all))

    with open('f_quantity_hydro.pk', 'wb') as f:
        #json.dump([f_quantity_all, f_hydro_all], f, ensure_ascii=False, indent=4)
        pickle.dump([f_quantity_all, f_hydro_all], f, pickle.HIGHEST_PROTOCOL)


'''

with open('f_climate.pk', 'rb') as f:
    date_list, f_climate_all = pickle.load(f)
with open('f_quantity_hydro.pk', 'rb') as f:
    f_quantity_all, f_hydro_all = pickle.load(f)


train_dates = []
dev_dates = []
test_dates = []
for date in date_list:
    if  date <= '19991231':
        train_dates.append(date)
    elif date <= '20001231':
        dev_dates.append(date)
    else:
        test_dates.append(date)

print(len(train_dates), len(dev_dates), len(test_dates))
seq_len = [1, 5, 10, 15, 25, 30, 60, 90, 180]


f_max = -np.ones((14,1))*np.inf
f_min = np.ones((14,1))*np.inf

for seqi in seq_len:
    print(seqi)
    train_sets = []
    dev_sets  =[]
    test_sets = []
    print('process train set', len(train_dates))
    
    i = 0
    while i < (len(train_dates) - seqi + 1):
        if i % 5000 == 0:
            print(str(i) + "/" + str(len(train_dates)))
        x_dates = train_dates[i:i+seqi]
        xs = []
        ys = []
        ds = []
        for date in x_dates:
            xc = f_climate_all[date]
            xq = f_quantity_all[date]
            yh = f_hydro_all[date]
            hukou = [yh[0]]
            yh = yh[1:5]
            x = np.concatenate([xc, xq, hukou])
            x = np.reshape(x, (-1, 1))
            f_max = np.amax(np.concatenate([f_max, x], axis=1), axis=1)
            f_max = np.reshape(f_max, (-1, 1))
            f_min = np.amin(np.concatenate([f_min, x], axis=1), axis=1)
            f_min = np.reshape(f_min, (-1, 1))
            xs.append(x)
            ys.append(yh)
            ds.append(date)
        train_sets.append([xs, ys, ds])
        i += 1
    print("dump trainset", len(train_sets))
    # mix-max norm
    train_sets_norm = []
    for xs_, ys, ds in train_sets:
        xs = []
        for x in xs_:
            x = (x - f_min) / (f_max - f_min)
            xs.append(x)
        train_sets_norm.append([xs, ys, ds])
    with open('train_seq_'+str(seqi)+'.pk', 'wb') as f:
        pickle.dump([train_sets, train_sets_norm, f_max, f_min], f, pickle.HIGHEST_PROTOCOL)
        

    i = 0
    print('process dev set', len(dev_dates))
    while i < (len(dev_dates) - seqi + 1):
        if i % 1000 == 0:
            print(str(i) + "/" + str(len(dev_dates)))
        x_dates = dev_dates[i:i+seqi]
        xs = []
        ys = []
        ds = []
        for date in x_dates:
            xc = f_climate_all[date]
            xq = f_quantity_all[date]
            yh = f_hydro_all[date]
            hukou = [yh[0]]
            yh = yh[1:5]
            x = np.concatenate([xc, xq, hukou])
            x = np.reshape(x, (-1, 1))
            xs.append(x)
            ys.append(yh)
            ds.append(date)
        dev_sets.append([xs, ys, ds])
        i += 1
    print("dump devset", len(dev_sets))
    # mix-max norm
    dev_sets_norm = []
    for xs_, ys, ds in dev_sets:
        xs = []
        for x in xs_:
            x = (x - f_min) / (f_max - f_min)
            xs.append(x)
        dev_sets_norm.append([xs, ys, ds])

    with open('dev_seq_'+str(seqi)+'.pk', 'wb') as f:
        pickle.dump([dev_sets, dev_sets_norm, f_max, f_min], f, pickle.HIGHEST_PROTOCOL)

    i = 0
    print('process test set', len(test_dates))
    while i < (len(test_dates) - seqi + 1):
        if i % 1000 == 0:
            print(str(i) + "/" + str(len(test_dates)))
        x_dates = test_dates[i:i+seqi]
        xs = []
        ys = []
        ds = []
        for date in x_dates:
            xc = f_climate_all[date]
            xq = f_quantity_all[date]
            yh = f_hydro_all[date]
            hukou = [yh[0]]
            yh = yh[1:5]
            x = np.concatenate([xc, xq, hukou])
            x = np.reshape(x, (-1, 1))
            xs.append(x)
            ys.append(yh)
            ds.append(date)
        test_sets.append([xs, ys, ds])
        i += 1
    print("dump testset", len(test_sets))
    # mix-max norm
    test_sets_norm = []
    for xs_, ys, ds in test_sets:
        xs = []
        for x in xs_:
            x = (x - f_min) / (f_max - f_min)
            xs.append(x)
        test_sets_norm.append([xs, ys, ds])

    with open('test_seq_'+str(seqi)+'.pk', 'wb') as f:
        pickle.dump([test_sets, test_sets_norm, f_max, f_min], f, pickle.HIGHEST_PROTOCOL)
        


