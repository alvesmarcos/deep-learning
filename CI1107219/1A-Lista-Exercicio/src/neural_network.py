import numpy as np
import generator as g

same = lambda x: x
# tupla(x,limit)
threshold = lambda x: np.where(x[0]>x[1], 1, 0) 

# Sigmoid: saída entre [0,1]
def sigmoid(z,derivate=False):
    return 1/(1+np.exp(-z))

# Tangente Hiperbólica: saída entre [-1,1]
def tanh(z,derivate=False):
    return (1-np.exp(-z))/(1+np.exp(-z))

# Retificadora (Relu): saída 0 caso entrada seja negativa e maior que 1 caso contrário
def relu(z,derivate=False):
    return np.where(z>0,z,0)

# Degrau: saída 0 se menor que 0 e saída 1 caso contrário
def degrau(z,derivate=False):
    return np.where(z>0,1,0)

class Layer:
    def __init__(self, units=1, activation=degrau, input_dim=1, use_bias=True):
        self.units = units
        self.weights = np.random.random((units, input_dim))-0.5
        self.bias = np.zeros(units)
        self.activation = activation
        self.input_dim = input_dim
        self.use_bias = use_bias
    
class NeuralNetwork:
    def __init__(self):
        self._layers = []
        self._learning_rate = 0.02
    
    def __backpropagation(self):
        pass

    def __error(self,y_i,y_pred):
        return y_i-y_pred
    
    def __forward(self, x_i):
        input_layer = x_i
        for layer in self._layers:
            z = np.dot(input_layer, layer.weights.T) + (layer.bias if layer.use_bias else np.zeros(layer.units))
            y = layer.activation(z)
        return y

    def __gradient(self, error):
        pass

    def __update_weight(self, error, x_i):
        self._layers[-1].weights = self._layers[-1].weights + self._learning_rate*np.dot(error.T, x_i)
        self._layers[-1].bias = self._layers[-1].bias + self._learning_rate*error[0]

    def add(self, layer):
        self._layers.append(layer)
    
    def evaluate(self,y_pred, y, func=same, dtype=int):
        score = 0
        total = 100.0/y.shape[0]
        # mesmo tipo de saida
        y_pred = y_pred.astype(dtype)
        y = y.astype(dtype)

        for y_i, y_pred_i in zip(y,y_pred):
            y_pred_i = func(y_pred_i)
            if np.array_equal(y_i,y_pred_i):
                score+=1
        return score*total

    def fit(self, X=None, Y=None, epochs=1, verbose=False): 
        for step in range(epochs):
            for x_i, y_i in zip(X,Y):
                # reshape x_i, y_i
                x_i =  x_i.reshape(1, X.shape[1])
                # propagacao para camadas da frente
                y_pred = self.__forward(x_i)
                error = self.__error(y_i,y_pred)
                # --
                self.__update_weight(error, x_i)

    def get_learning_rate(self):
        return self._learning_rate

    def get_weights(self):
        return self._layers[-1].weights, self._layers[-1].bias

    def predict(self, X, verbose=False):
        y_pred = []
        for i in range(len(X)):
            y_pred.append(self.__forward(X[i]))
        return np.array(y_pred)

if __name__=='__main__':
    x,y = g.data_1A1(1000,dtype='train')
    D = x.shape[1]
    model = NeuralNetwork()
    model.add(Layer(units=8, activation=degrau, input_dim=D))
    model.fit(x, y, epochs=500, verbose=False)
    # conjunto de validacao
    z,l = g.data_1A1(500,dtype='validation')

    w,b = model.get_weights()
    y_pred = model.predict(z, verbose=False)
    accurancy = model.evaluate(y_pred, l, func=same)

    print('w:', w)
    print('b:', b)
    print('Accurancy: {0:.1f}%'.format(accurancy))

    #reset base de dados
    # g.data_reset_1A1('train',True)
    # g.data_reset_1A1('validation',True)
    