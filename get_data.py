# 当日データの取得
from bs4 import BeautifulSoup    #importする
import requests
import pandas as pd
import numpy as np
from datetime import date

def get_data(date, race_number, stadium):
    '''
    ボートレースホームページからデータを取得する関数
    int race_number: レース番号
    str stadium: 競艇場
    date date: 日付

    return:
    pandas.DataFrame: データフレーム
    '''
    date = date.strftime("%Y%m%d")
    ## Web頁を取得して解析する
    load_url = "https://www.boatrace.jp/owpc/pc/race/racelist?rno=" + str(race_number) + "&jcd=" + str(stadium) + "&hd=" + str(date)
    html = requests.get(load_url)

    soup = BeautifulSoup(html.content, "html.parser")    # HTML を　解析
    if soup is None:
        return 0, 0, 0

    # print(soup.find_all(class_ = "is-fs11"))
    racers = soup.find_all(class_ = "is-fs11")
    # データの取得
    tei_list = [1, 2, 3, 4, 5, 6]
    racer_list = []
    moter_list = [] #
    boat_list = [] #
    tenji_time_list = [] #
    sinnyu_list = [1, 2, 3, 4, 5, 6]
    huusoku_list = [] # 
    height_wave_list = [] #
    age_list = []  # 
    weight_list = []  #
    kyu_list = []  #
    zenkoku_rate_list = [] # 
    zenkoku2ren_rate_list = [] # 
    touchi_rate_list = [] # 
    touchi2ren_rate_list = []# 
    moter2ren_list = [] #
    boat2ren_list = [] #

    # tyokuzen_soup = soup.find(class_ = "tab3 is-type1__3rdadd")
    # print(tyokuzen_soup.find_all("a")[1].get("href"))
    boat_url = "https://www.boatrace.jp/"
    tyokuzen_url = "https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno=" + str(race_number) + "&jcd=" + str(stadium) + "&hd=" + str(date)
    # ozz_href = tyokuzen_soup.find_all("a")[0].get("href")
    # tyokuzen_href = tyokuzen_soup.find_all("a")[1].get("href")
    # tyokuzen_url = urllib.parse.urljoin(boat_url, tyokuzen_href)
    print(tyokuzen_url)

    # 直前情報の取得
    tyokuzen_html = requests.get(tyokuzen_url)
    tyokuzen_soup = BeautifulSoup(tyokuzen_html.content, "html.parser")    # HTML を　解析
    tenji_soup = tyokuzen_soup.find_all(class_ = "is-fs12")

    racers = soup.find_all(class_ = "is-fs11")
    rate_soup = soup.find_all(class_ = "is-lineH2")

    for i in range(6):
        racer_list.append(int(racers[2 * i + 2].text[0:4]))
        kyu_list.append(racers[2 * i + 2].get_text(",").split(",")[1])
        age_list.append(int(racers[2 * i + 3].get_text(",").split(",")[1][0:2]))
        weight_list.append(float(racers[2 * i + 3].get_text(",").split(",")[1][4:7]))
        zenkoku_rate_list.append(float(rate_soup[5 * i + 1].get_text(",").split(",")[0]))
        zenkoku2ren_rate_list.append(float(rate_soup[5 * i + 1].get_text(",").split(",")[1]))
        touchi_rate_list.append(float(rate_soup[5 * i + 2].get_text(",").split(",")[0]))
        touchi2ren_rate_list.append(float(rate_soup[5 * i + 2].get_text(",").split(",")[1]))
        moter_list.append(int(rate_soup[5 * i + 3].get_text(",").split(",")[0]))
        moter2ren_list.append(float(rate_soup[5 * i + 3].get_text(",").split(",")[1]))
        boat_list.append(int(rate_soup[5 * i + 4].get_text(",").split(",")[0]))
        boat2ren_list.append(float(rate_soup[5 * i + 4].get_text(",").split(",")[1]))
        tenji_time_list.append(float(tenji_soup[i].get_text(",").split(",")[9]))
        # tenki_list.append(tyokuzen_soup.find_all(class_ = "weather1_bodyUnitLabelTitle")[1].text + " ")
        # huusoku_list.append(int(tyokuzen_soup.find_all(class_ = "weather1_bodyUnitLabelData")[1].text.replace("m", "")))
        # height_wave_list.append(int(tyokuzen_soup.find_all(class_ = "weather1_bodyUnitLabelData")[3].text.replace("cm", "")))
        sinnyu_list[int(tyokuzen_soup.find_all(class_ = "table1_boatImage1Number")[i].text)-1] = i + 1
    
    # データセットを作る
    df = pd.DataFrame([tenji_time_list + zenkoku_rate_list + zenkoku2ren_rate_list + touchi_rate_list + touchi2ren_rate_list + moter2ren_list + boat2ren_list])
    
    # 標準化
    x_mean = np.array([ 6.78542098,  6.79765159,  6.80379114,  6.806126  ,  6.80950275,
        6.80931456,  5.60533807,  5.35570699,  5.344113  ,  5.24298133,
        5.02964286,  4.91359575, 37.00515338, 34.19676115, 34.07851494,
       33.14710122, 31.06553118, 30.15363206,  5.33100376,  5.03872199,
        5.0255756 ,  4.8980676 ,  4.65174962,  4.50528632, 35.29574486,
       32.19541658, 32.08059573, 31.10267014, 28.98300337, 28.01293559,
       32.44684387, 32.36630935, 32.34732691, 32.34510236, 32.33181621,
       32.32493411, 32.14672585, 32.12760376, 32.13931031, 32.1055591 ,
       32.14505162, 32.12722994])
    
    x_std = np.array([ 0.11169902,  0.11147992,  0.11193437,  0.11214193,  0.11201465,
        0.11241692,  1.13989147,  1.15812954,  1.16580053,  1.30348026,
        1.44075214,  1.5899491 , 13.17228904, 13.31942175, 13.37168358,
       14.35741312, 15.15872694, 16.13317934,  1.76689179,  1.80308072,
        1.80998455,  1.93378164,  2.05379401,  2.18330446, 17.79482228,
       17.7708883 , 17.81324487, 18.49246876, 18.8888694 , 19.50463936,
       11.52354525, 11.49856684, 11.4781225 , 11.47716526, 11.4691367 ,
       11.48470464, 10.92301621, 10.89902774, 10.9074183 , 10.89644409,
       10.92766482, 10.91947778])
    df = (df - x_mean) / x_std
    
    # one-hot encoding
    class_mapping = {'B2':1,'B1':2,'A2':3,'A1':4}
    for i in range(6):
        kyu_list[i] = class_mapping[kyu_list[i]]
    onehot_cols = ['stadium_' + str(i) for i in range(1, 25)] + ["級別_" + str(i) + "_" + str(j) for i in range(1, 7) for j in range(1, 5)]
    
    
    X_onehot = pd.DataFrame(np.zeros((1, len(onehot_cols)), dtype=int), columns=onehot_cols)
    X_onehot["stadium_" + str(int(stadium))] = 1
    for i in range(6):
        X_onehot["級別_" + str(i + 1) + "_" + str(kyu_list[i])] = 1
    X_onehot = X_onehot.fillna(0)
    df = pd.concat([df, X_onehot], axis=1)
    return df

if __name__ == "__main__":
    get_data(date.today(), '01', 11)
