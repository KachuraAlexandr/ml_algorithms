import numpy as np
import scipy
from scipy.special import expit
from scipy import sparse

 
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
        margin = y_zero_to_minus * X.dot(w.reshape((-1, 1)))
        return np.sum(np.logaddexp(0, -margin)) / X.shape[0] + 0.5 * self.l2_coef * np.inner(w, w)
        
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
        margin = X.dot(w.reshape((-1, 1))) * y_zero_to_minus
        return -(expit(-margin) * X * y_zero_to_minus).sum(axis=0) / X.shape[0] + self.l2_coef * w
