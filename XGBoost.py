import pandas as pd
#read data
data_path="Intermediate_ml\melb_data.csv"
data=pd.read_csv(data_path)
from sklearn.model_selection import train_test_split
#select the predectore
cols_to_use=['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X=data[cols_to_use]
y=data['Price']
#sperate data into train and testing 
X_train,X_valid,y_train,y_valid=train_test_split(X,y,random_state=0)
"""
Dalam contoh ini, Anda akan bekerja dengan pustaka XGBoost. XGBoost adalah singkatan dari peningkatan gradien ekstrem,
yang merupakan implementasi peningkatan gradien dengan beberapa fitur tambahan yang berfokus pada kinerja dan kecepatan. 
(Scikit-learn memiliki versi lain dari peningkatan gradien, tetapi XGBoost memiliki beberapa keunggulan teknis.)
Di sel kode berikutnya, kita mengimpor scikit-learn API untuk XGBoost (xgboost.XGBRegressor).
Ini memungkinkan kita untuk membangun dan menyesuaikan model seperti yang kita lakukan di scikit-learn.
Seperti yang akan Anda lihat di output, kelas XGBRegressor memiliki banyak parameter yang dapat disetel -- Anda akan segera mempelajarinya!
"""
#preprocessing
from sklearn.impute import SimpleImputer
preprocessing=SimpleImputer(strategy='constant')
#define model
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
my_model=XGBRegressor()
my_pipeline=Pipeline(steps=[
    ('preprocessing',preprocessing),
    ('model',my_model)
])
my_model=my_model.fit(X_train,y_train)
predicts=my_model.predict(X_valid)
from sklearn.metrics import mean_absolute_error
MAE=mean_absolute_error(y_valid,predicts)
print("Hasil predict:",predicts)
print("MAE:",MAE)#239431.96930228276

#Parameter Tuning

#n_estimator
"""
digunakan untnuk berapa kali proses melalui siklus pada ensamble atau
berapa jumalah model yang dimasukan ke dalam ensamble
-nilai yang rendah=>underfitting->model predictsi tidak akurat pada data pelatihan dan validasi(testing)
-nilai yang tinggi=>overfitting->model predictsi akurat pada data pelatihan namun pada data validasi tidak akurat

nilai tipikal yang biasa digunakan adalah bersikar antara 100-1000 yang bergantung pada parameter learning rate-nya

defaultnya n_estimators=0
"""
mymodel2=XGBRegressor(n_estimators=500)
print(mymodel2)
mymodel2=mymodel2.fit(X_train,y_train)
predicts2=mymodel2.predict(X_valid)
MAE2=mean_absolute_error(y_valid, predicts2)
print(MAE2)#249306.81565434003->makin besr MAEnya

#early_stopping_rounds
"""
->secara otomatis untuk menemukan nomor ideal n_estimators 
digunakan untuk pemberhentian awal pengulangan penambahan model pada ensamble atau
penghentian lebih awal akan menyebabkan 
model berhenti mengulangi saat skor validasi berhenti meningkat
tips:
set n_estimator yang tinggi namun gunakan early_stopping_rounds untuk menemukan waktu optimal dalam pemberhentian iterasi
->tentukan berapa nilai untuk berapa putaran penurunan lurus yang di izinkan sebelum berhenti
anggap early_stopping_rounds=5 yang berarti setelah lima menghasilkan skor validasi yang memburuk


saat menghitung sebuah early_stopping_rounds, kita perlu juga menyisihkan beberapa data untuk menghitung skor validasi
=>stell parameter eval_set

"""
mymodel2=mymodel2.fit(X_train,y_train,
                      early_stopping_rounds=10,#7++->KONSTANT =>240930.0053939617
                      eval_set=[(X_valid,y_valid)],
                      verbose=False
                      )
print(mymodel2)
predicts3=mymodel2.predict(X_valid)
print(predicts3)
print(mean_absolute_error(y_valid, predicts3))
#245079.02440629603->5
#240930.0053939617-.10
#240930.0053939617->30
#240930.0053939617->100
#240930.0053939617->250


#learning_rate
"""
learning_rate default=0.01
digunakan dalam pengalian pada n_estimator dengan nilai learning ratenya hal ini ditunjukan
untuk mencegah terjadinya overfitting pada saat mengeset nilai n_estimators yang besar, maka
tips:
set n_estimators yang besar untuk learning_rate yang kecil
=>menghasilkan modul yang lebih akurat
n_jobs=>untuk mengatur core yang diberikan pada ensamble di setiap siklus eksekusi proses model yang diberikan pada ensamble dalam iteratif
"""
my_model3=XGBRegressor(n_estimators=500,learning_rate=0.05,n_jobs=5)
my_model3=my_model3.fit(X_train,y_train,
                        early_stopping_rounds=10,
                        eval_set=[(X_valid,y_valid)],
                        verbose=False
                        )
predicts4=my_model3.predict(X_valid)
print(predicts4)
print(mean_absolute_error(y_valid, predicts4))
#243355.74739506628
