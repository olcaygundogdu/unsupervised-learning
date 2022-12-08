##############################################
# Gözetimsiz Öğrenme ile Müşteri Segmentasyonu
##############################################

# İş Problemi: FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre gruplar oluşturulacak.

# Veri Seti Hikayesi
# Veri seti Flo’dan son alışverişlerini 2020 -2021 yıllarında Omni Channel(hem online hem offline alışverişyapan) olarak
# yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi
# store_type: 3 farklı companyi ifade eder. A company'sinden alışveriş yapan kişi B'dende yaptı ise A, B şeklinde yazılmıştır

import datetime as dt
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

def check_outlier(dataframe, col_name, q1=0.20, q3=0.80):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

# GÖREV 1: VERİYİ HAZIRLAMA

# Adım 1: flo_data_20K.csv verisini okutunuz.

df_read = pd.read_csv("Week 9 (Yapay Öğrenme III)/Bitirme Projesi IV/flo_data_20K.csv")
df = df_read.copy()
df.head()
df.info()
df.shape

# Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.
# Not: Tenure (Müşterinin yaşı), Recency (enson kaç gün önce alışveriş yaptığı) gibi yeni değişkenler oluşturabilirsiniz.
date = [col for col in df.columns if "date" in col]
df[date] = df[date].apply(pd.to_datetime)

df["last_order_date"].max()   # 2021-05-30
today_date = dt.datetime(2021, 6, 1)

df["NEW_TOTAL_ORDER"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["NEW_TOTAL_PRİCE"] = df["customer_value_total_ever_online"] + df["customer_value_total_ever_offline"]

for i in range(0, df.shape[0]):
    df.loc[i, "NEW_TENURE"] = (today_date - df.loc[i, "first_order_date"]).days
for i in range(0, df.shape[0]):
    df.loc[i, "NEW_RECENCY"] = (today_date - df.loc[i, "last_order_date"]).days

# df["last_order_date"].apply(lambda x: (today_date - x).days)

id = df["master_id"]

df = df.drop(date, axis=1)
df = df.drop(["master_id", "interested_in_categories_12"], axis=1)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

df.describe().T

df = one_hot_encoder(df, cat_cols)

# GÖREV 2: K-MEANS İLE MÜŞTERİ SEGMENTASYONU

# Adım 1: Değişkenleri standartlaştırınız.

scaled = StandardScaler().fit_transform(df[num_cols])
df[num_cols] = pd.DataFrame(scaled, columns=df[num_cols].columns)
df.head()

# Adım 2: Optimum küme sayısını belirleyiniz.

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_    # k = 7

# Adım 3: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.

kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df)

clusters_kmeans = kmeans.labels_
df["CLUSTER"] = clusters_kmeans
df["CLUSTER"] = df["CLUSTER"] + 1

df.head()

df_km_final = df_read
df_km_final["CLUSTER"] = clusters_kmeans
df_km_final["CLUSTER"] = df_km_final["CLUSTER"] + 1
df_km_final.head()

# Adım 4: Her bir segmenti istatistiksel olarak inceleyeniz.

df_km_final.groupby("CLUSTER").agg(["count", "mean", "median"])

# GÖREV 3: HİERARCHİCAL CLUSTERİNG İLE MÜŞTERİ SEGMENTASYONU

# Adım 1: Görev 2'de standırlaştırdığınız dataframe'i kullanarak optimum küme sayısını belirleyiniz.

hc_average = linkage(df, "average")

plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=7.5, color='r', linestyle='--')
plt.show()

# 7.60

# Adım 2: Modelinizi oluşturunuz ve müşterilerinizi segmentleyiniz.

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=4, linkage="average")

clusters = cluster.fit_predict(df)

df["CLUSTER"] = clusters

df["CLUSTER"] = df["CLUSTER"] + 1

df.head()

# Adım 3: Her bir segmenti istatistiksel olarak inceleyeniz.

df.groupby("CLUSTER").agg(["count", "mean", "median"])