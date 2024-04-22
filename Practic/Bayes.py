import numpy as np
from collections import OrderedDict
from functools import reduce
from operator import mul


class Gauss:
    def fit(self, X: np.array, y: np.array) -> 'Gauss':
        '''
        Funcția de potrivire a modelului gauss naive bayes.
        :param X: np.array
        Matricea de caracteristici.
        :param y: np.array
        Vectorul țintă.
        :return: Gauss
        Această funcție returnează obiectul modelului antrenat.
        '''
        # Pregătirea vectorilor pentru medie, deviație standard și priors.
        self.mu = OrderedDict() # Ce e aceasta
        self.std = OrderedDict() # ce e aceasta
        self.priors = OrderedDict() # ce e aceasta
        # Calcularea vectorilor pentru medie, deviație standard și priors pentru fiecare clasă.
        for cls in np.unique(y):
            self.priors[cls] = len(y[y == cls]) / len(y)
            self.mu[cls] = np.mean(X[np.where(y == cls)], axis=0)
            self.std[cls] = np.std(X[np.where(y == cls)], axis=0)
        return self

    # În listarea de mai sus calculăm priors pentru că le vom folosi în faza de prezicere.

    def normal_distribution(self, x: np.array, cls: str, i: int) -> float:
        '''
        Formula distribuției normale.
        :param x: np.array
        Eșantionul pentru care dorim să găsim probabilitatea.
        :param cls: str sau int
        Clasa pentru care dorim să calculăm probabilitatea.
        :param i: int
        Indexul caracteristicii.
        :return: float
        Probabilitatea eșantionului pentru distribuția normală.
        '''
        exponent = np.exp(-((x[i] - self.mu[cls][i]) ** 2))
        exponent /= (2 * self.std[cls][i] ** 2)
        return (1 / (np.sqrt(2 * np.pi) * self.std[cls][i] ** 2)) * exponent

    # Acum să explorăm funcția predict_proba. Este responsabilă pentru calcularea probabilității fiecărui eșantion de a aparține fiecărei clase folosind formula din a doua definiție a Teoremei lui Naive Bayes.

    def predict_proba(self, X: np.array) -> np.array:
        '''
        Această funcție returnează probabilitatea pentru fiecare eșantion din setul de date.
        :param X: np.array
        Matricea de caracteristici folosită pentru a face predicții.
        :return: np.array
        Un tablou cu probabilitățile pentru fiecare clasă pentru fiecare eșantion.
        '''
        # Crearea unei liste goale cu probabilități.
        y_pred = []
        # Calcularea probabilităților pentru fiecare clasă pentru fiecare eșantion.
        for x in X:
            y_pred.append([])
            # Calcularea probabilității pentru fiecare clasă pentru acest eșantion.
            for cls in self.priors:
                prob = reduce(mul, [self.normal_distribution(x, cls, i) for i in range(len(x))]) * self.priors[cls]
                y_pred[-1].append(prob)
            y_pred[-1] = np.array(y_pred[-1])
            # Normalizarea vectorului.
            y_pred[-1] = y_pred[-1] / np.linalg.norm(y_pred[-1])
        return np.array(y_pred)

    # Această funcție la sfârșit normalizează fiecare vector de probabilitate pentru că rezultatul formulei Naive Bayes nu este între 0 și 1.

    # Acum că putem calcula probabilitățile, le putem folosi pentru a prezice clasele fiecărui eșantion:

    def predict(self, X: np.array) -> np.array:
        '''
        Această funcție returnează clasa prezisă pentru fiecare eșantion din setul de date.
        :param X: np.array
        Matricea de caracteristici folosită pentru a face predicții.
        :return: np.array
        Un tablou cu clasele prezise pentru fiecare eșantion.
        '''
        # Crearea unei liste goale pentru stocarea claselor prezise.
        y = []
        # Obținerea probabilităților prezise pentru fiecare eșantion.
        probas = self.predict_proba(X)
        # Obținerea clasei cu cea mai mare probabilitate pentru fiecare eșantion.
        for pr in probas:
            y.append(list(self.priors.keys())[np.argmax(pr)])
        return y