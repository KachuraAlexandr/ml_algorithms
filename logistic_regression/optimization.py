import numpy as np
import scipy
from scipy.special import expit
from scipy import sparse
import time
 
 
class BaseSmoothOracle:
    """
    Базовый класс для реализации оракулов.
    """
    def func(self, w):
        """
        Вычислить значение функции в точке w.
        """
        raise NotImplementedError('Func oracle is not implemented.')
 
    def grad(self, w):
        """
        Вычислить значение градиента функции в точке w.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
 
        
class BinaryLogistic(BaseSmoothOracle):
    """
    Оракул для задачи двухклассовой логистической регрессии.
    
    Оракул должен поддерживать l2 регуляризацию.
    """
    
    def __init__(self, l2_coef):
        """
        Задание параметров оракула.
        
        l2_coef - коэффициент l2 регуляризации
        """
        self.l2_coef = l2_coef
     
    def func(self, X, y, w):
        """
        Вычислить значение функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        y_zero_to_minus = y.copy()
        y_zero_to_minus[y == 0] = -1
        y_zero_to_minus = y_zero_to_minus.reshape((-1, 1))
        margin = np.array((scipy.sparse.csr_matrix(X) * w.reshape((-1, 1))).\
          multiply(y_zero_to_minus.reshape((-1, 1)))).ravel()
        return np.sum(np.logaddexp(0, -margin)) / X.shape[0] + \
          0.5 * self.l2_coef * np.inner(w, w)
        
    def grad(self, X, y, w):
        """
        Вычислить градиент функционала в точке w на выборке X с ответами y.
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w - одномерный numpy array
        """
        y_zero_to_minus = y.copy()
        y_zero_to_minus[y == 0] = -1
        y_zero_to_minus = y_zero_to_minus.reshape((-1, 1))
        margin = np.array((scipy.sparse.csr_matrix(X) * w.reshape((-1, 1))).\
          multiply(y_zero_to_minus.reshape((-1, 1)))).ravel()
        return -np.array(scipy.sparse.csr_matrix(X).multiply(expit(-margin)).\
          multiply(y_zero_to_minus).sum(axis=0)).ravel() / X.shape[0] + self.l2_coef * w
 
 
class GDClassifier:
    """
    Реализация метода градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    
    def __init__(self, loss_function, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=1000, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
                
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию.
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход 
        
        max_iter - максимальное число итераций     
        
        **kwargs - аргументы, необходимые для инициализации   
        """
        self.loss_function = loss_function
        if loss_function == 'binary_logistic':
            l2_coef = 0
            if 'l2_coef' in kwargs:
                l2_coef = kwargs['l2_coef']
            self.loss_oracle = BinaryLogistic(l2_coef)
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance 
        self.max_iter = max_iter
        
        
    def fit(self, X, y, w_0=None, trace=False):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
        
        w_0 - начальное приближение в методе
        
        trace - переменная типа bool
      
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Длина словаря history = количество итераций + 1 (начальное приближение)
        
        history['time']: list of floats, содержит интервалы времени между двумя итерациями метода
        history['func']: list of floats, содержит значения функции на каждой итерации
        (0 для самой первой точки)
        """
        if w_0 is None:
            w_0 = np.zeros(X.shape[1])
        loss_func = self.loss_oracle.func
        loss_grad = self.loss_oracle.grad
        step_alpha = self.step_alpha
        step_beta = self.step_beta
        tolerance = self.tolerance
        max_iter = self.max_iter
        prev_time = time.time()
        loss_old_val = loss_func(X, y, w_0)
        learning_rate = step_alpha # there is no need to divide it on i ** step_beta because i == 1
        w = w_0 - learning_rate * loss_grad(X, y, w_0)
        loss_new_val = loss_func(X, y, w)
        cur_time = time.time()
        history = {'time': [0, cur_time - prev_time], 'func': [loss_old_val, loss_new_val]}
        i = 2
        while i <= max_iter and abs(loss_new_val - loss_old_val) >= tolerance:
            learning_rate = step_alpha / (i ** step_beta)
            w -= learning_rate * loss_grad(X, y, w)
            loss_old_val = loss_new_val
            loss_new_val = loss_func(X, y, w)
            prev_time = cur_time
            cur_time = time.time()
            history['time'].append(cur_time - prev_time)
            history['func'].append(loss_new_val)
            i += 1
        self.weights = w
        if trace:
            return history
 
    def predict(self, X):
        """
        Получение меток ответов на выборке X
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: одномерный numpy array с предсказаниями
        """
        return scipy.sparse.csr_matrix(X) * self.weights > 0
 
    def predict_proba(self, X):
        """
        Получение вероятностей принадлежности X к классу k
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        return: двумерной numpy array, [i, k] значение соответветствует вероятности
        принадлежности i-го объекта к классу k 
        """
        if loss_function == "binary_logistic":
            proba_arr = np.zeros(X.shape[0])
            proba_arr = scipy.special.expit(np.dot(self.weight,X.transpose))
        return proba_arr
        
    def get_objective(self, X, y):
        """
        Получение значения целевой функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: float
        """
        return self.loss_oracle(X, y, self.weights)
        
    def get_gradient(self, X, y):
        """
        Получение значения градиента функции на выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        y - одномерный numpy array
        
        return: numpy array, размерность зависит от задачи
        """
        return self.loss_oracle(X,y, self.weights)
    
    def get_weights(self):
        """
        Получение значения весов функционала
        """    
        return self.weights
 
 
class SGDClassifier(GDClassifier):
    """
    Реализация метода стохастического градиентного спуска для произвольного
    оракула, соответствующего спецификации оракулов из модуля oracles.py
    """
    
    def __init__(self, loss_function, batch_size, step_alpha=1, step_beta=0, 
                 tolerance=1e-5, max_iter=1000, random_seed=153, **kwargs):
        """
        loss_function - строка, отвечающая за функцию потерь классификатора. 
        Может принимать значения:
        - 'binary_logistic' - бинарная логистическая регрессия
        
        batch_size - размер подвыборки, по которой считается градиент
        
        step_alpha - float, параметр выбора шага из текста задания
        
        step_beta- float, параметр выбора шага из текста задания
        
        tolerance - точность, по достижении которой, необходимо прекратить оптимизацию
        Необходимо использовать критерий выхода по модулю разности соседних значений функции:
        если |f(x_{k+1}) - f(x_{k})| < tolerance: то выход 
        
        
        max_iter - максимальное число итераций (эпох)
        
        random_seed - в начале метода fit необходимо вызвать np.random.seed(random_seed).
        Этот параметр нужен для воспроизводимости результатов на разных машинах.
        
        **kwargs - аргументы, необходимые для инициализации
        """
        self.loss_function = loss_function
        if loss_function == 'binary_logistic':
            self.loss_oracle = oracles.BinaryLogistic(0)
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance 
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_seed = random_seed
        
    def fit(self, X, y, w_0=None, trace=False, log_freq=1):
        """
        Обучение метода по выборке X с ответами y
        
        X - scipy.sparse.csr_matrix или двумерный numpy.array
        
        y - одномерный numpy array
                
        w_0 - начальное приближение в методе
        
        Если trace = True, то метод должен вернуть словарь history, содержащий информацию 
        о поведении метода. Если обновлять history после каждой итерации, метод перестанет 
        превосходить в скорости метод GD. Поэтому, необходимо обновлять историю метода лишь
        после некоторого числа обработанных объектов в зависимости от приближённого номера эпохи.
        Приближённый номер эпохи:
            {количество объектов, обработанных методом SGD} / {количество объектов в выборке}
        
        log_freq - float от 0 до 1, параметр, отвечающий за частоту обновления. 
        Обновление должно проиходить каждый раз, когда разница между двумя значениями приближённого номера эпохи
        будет превосходить log_freq.
        
        history['epoch_num']: list of floats, в каждом элементе списка будет записан приближённый номер эпохи:
        history['time']: list of floats, содержит интервалы времени между двумя соседними замерами
        history['func']: list of floats, содержит значения функции после текущего приближённого номера эпохи
        history['weights_diff']: list of floats, содержит квадрат нормы разности векторов весов с соседних замеров
        (0 для самой первой точки)
        """
        np.random.seed(self.random_seed)
        if w_0 is None:
            w_0 = np.zeros(X.shape[1])
        loss_func = self.loss_oracle.func
        loss_grad = self.loss_oracle.grad
        step_alpha = self.step_alpha
        step_beta = self.step_beta
        tolerance = self.tolerance
        max_iter = self.max_iter
        train_size = X.shape[0]
        history = {'epoch_num': 0.0, 'time': 0.0, 'func': 0.0, 'weights_diff': 0.0}
        log_period = round(log_freq * train_size)
        loss_old_val = 0.0
        loss_new_val = loss_func(X, y, w_0)
        batch_size = self.batch_size
        w_new = w_0
        i = 1
        cur_time = time.time()
        while i <= max_iter and abs(loss_new_val - loss_old_val) >= tolerance:
            shuffled_indexes = np.random.permutation(train_size)
            batch_start = 0
            while batch_start < train_size:
                learning_rate = step_alpha / (i ** step_beta)
                w = w - learning_rate * \
                  loss_grad(X[batch_start:batch_start + batch_size, :], \
                                  y[batch_start:batch_start + batch_size], w)
                loss_old_val = loss_new_val
                loss_new_val = loss_func(X[batch_start:batch_start + batch_size, :], \
                                                        y[batch_start:batch_start + batch_size], w)
                if i % log_period == 0:
                    prev_time = cur_time
                    cur_time = time.time()
                    history['epoch_num'].append(i / train_size)
                    history['time'].append(cur_time - prev_time)
                    history['func'].append(loss_new_val)
                    w_old = w_new
                    w_new = w
                    w_diff = w_new - w_old
                    history['weights_diff'].append(np.inner(w_diff, w_diff))
                batch_start += batch_size
                i += 1
        self.weights = w
        if trace:
            return history
