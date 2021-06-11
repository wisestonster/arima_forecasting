import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('data/btc2021.csv')  # 업비트에서 가져온 데이터
df.sort_values(by=['Date'], axis=0, inplace=True)
df.set_index('Date', inplace=True)
df.plot()
plt.xlabel('day')
plt.ylabel('price')
plt.show()

from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

# (AR=2, 차분=1, MA=2) 파라미터로 ARIMA 모델을 학습합니다.
model = ARIMA(df.open.values, order=(2, 1, 2))

# trend : constant를 가지고 있는지, c - constant / nc - no constant
# disp : 수렴 정보를 나타냄
model_fit = model.fit(trend='c', full_output=True, disp=True)
print(model_fit.summary())

forecast_data = model_fit.forecast(steps=5)  # 학습 데이터셋으로부터 5일 뒤를 예측합니다.
pred_y = forecast_data[0].tolist()  # 마지막 5일의 예측 데이터입니다. (2018-08-27 ~ 2018-08-31)
test_y = df.open.values  # 실제 5일 가격 데이터입니다. (2018-08-27 ~ 2018-08-31)
pred_y_lower = []  # 마지막 5일의 예측 데이터의 최소값입니다.
pred_y_upper = []  # 마지막 5일의 예측 데이터의 최대값입니다.
for lower_upper in forecast_data[2]:
    lower = lower_upper[0]
    upper = lower_upper[1]
    pred_y_lower.append(lower)
    pred_y_upper.append(upper)

plt.plot(pred_y, color="gold")  # 모델이 예상한 가격 그래프입니다.
plt.plot(pred_y_lower, color="red")  # 모델이 예상한 최소가격 그래프입니다.
plt.plot(pred_y_upper, color="blue")  # 모델이 예상한 최대가격 그래프입니다.
# plt.plot(test_y, color="green") # 실제 가격 그래프입니다.
plt.show()

fore = model_fit.forecast(steps=5)
x = np.arange(0, 5)
y = fore[0]
plt.plot(x, y)
plt.show()

print('----- 향후 5일간 단기 가격 예측')
for i in range(len(y)):
    print("D+{} = {}".format(i, y[i]))