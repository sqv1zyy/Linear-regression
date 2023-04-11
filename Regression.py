import sklearn
from sklearn.datasets import load_boston
boston = load_boston()
print(boston.DESCR)
#Разбиваем данные на тестовые и тренировочные
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=42)
#Обучаем модель
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
#Предсказывание
y_pred = model.predict(X_test)
p = y_pred.shape
print(p)
#Сравнение результатов
print(y_pred[0], y_test[0])
#найдем среднею абсолютную ошибку
from sklearn.metrics import mean_absolute_error
a = mean_absolute_error(y_test,y_pred)
print(a)