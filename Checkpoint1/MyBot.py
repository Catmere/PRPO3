from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas as pd

class MYBOT:
    def __init__(self, myShop):
        # Конструктор для бота
        # получаем тренировочные данные
        self.dataset=myShop.getTrainingData()
        # начальные значения параметров
        self.age=1
        self.sex = 1
        self.sex = 0
        self.address=1
        self.category=1
        self.price=100
        self.color=1
        self.season=1
        self.material=1

    def botTraining(self, isPrint):
        # обучение бота
        # массив значений
        array = self.dataset.values
        # Числовые данные
        X = array[:, 1:9]
        # Названия товаров
        Y = array[:, 0]
        # Размер проверочной выборки 20% от всех данных
        validation_size = 0.20
        #Указывает, что выбор случайных данных должен быть одинаковым при каждом вызове обучения
        seed = 7
        #Разделение данных на тренировочные и проверочные
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                        random_state=seed, shuffle=True)

        seed = 7
        scoring = 'accuracy'
        # Кросс-валидация K-fold
        kfold = model_selection.KFold(n_splits=10, random_state=seed, shuffle=True)
        # Проверка полученных моделей с помощью скользящего контроля
        cv_results = model_selection.cross_val_score(KNeighborsClassifier(), X_train, Y_train, cv=kfold,
                                                     scoring=scoring)
        #Среднее значение и среднеквадратичное отклонение
        msg = "%s: %f (%f)" % ('KNN', cv_results.mean(), cv_results.std())

        #создаем модель K-ближайших соседей
        self.knn = KNeighborsClassifier()
        #обучаем модель
        self.knn.fit(X_train, Y_train)
        #проверяем качество обученной модели на тестовых данных
        predictions = self.knn.predict(X_validation)
        #Вывод на экран идет по запросу пользователя
        if isPrint==1:
            #размер входных данных
            print(self.dataset.shape)
            # Первые 20 строк
            print(self.dataset.head(20))

            # Описание входных данных
            print(self.dataset.describe())
            #Группировка данных по названию товара
            print(self.dataset.groupby('name').size())
            # Оценка качества модели
            print(msg)
            #Средняя ошибка распознавания
            print(accuracy_score(Y_validation, predictions))
            #Количество распознанных товаров по видам
            print(confusion_matrix(Y_validation, predictions))
            #сводная таблица распределения вероятностей распознавания товаров
            print(classification_report(Y_validation, predictions))

    def getUserChoice(self):
        #Ввод данных пользователя
        self.age=int(input("Введите ваш возраст: "))
        s=input("Введите ваш пол (M - муж., F - жен): ")
        if s=='M':
            self.sex=1
        else:
            self.sex=0
        self.address = int(input("Введите Ваш город (1 - Москва, 2 - Санкт-Петербург): "))
        self.category = int(input("Введите категорию товаров (1 - Ассортимент, 2 - Одежда, 3 - Обувь, 4 - Аксессуары): "))
        self.price = int(input("Введите цену: "))
        self.color = int(input("Введите цвет (1 - Белый, 2 - Черный, 3 - Синий, 4 - Смешанный): "))
        self.season = int(input("Введите сезон (1 - Зима, 2 - Лето, 3 - Осень, 4 - Весна): "))
        self.material = int(input("Введите материал (1 - Хлопок, 2 - Лен, 3 - Синтетика, 4  - Кожа): "))
        #Возвращается двумерный массив с данными
        return [[self.category,self.price,self.color,self.season,self.material,self.age,self.sex,self.address]]

    def getPrecigion(self,sampleData):
        #Подбор товара по данным пользователя
        #Распознавание наиболее подходящего товара по данным пользователя
        prediction = self.knn.predict(sampleData)
        return prediction[0]
