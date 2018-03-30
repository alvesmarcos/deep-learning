import numpy as np
import generator as g

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
        self.weights = np.random.random((1, input_dim))-0.5
        self.bias = 0
        self.activation = activation
        self.input_dim = input_dim
        self.use_bias = use_bias
    
class NeuralNetwork:
    def __init__(self):
        self._layers = []
        self._learning_rate = 0.2
    
    def __backpropagation(self):
        pass
    
    def __forward(self, x_i):
        input_layer = x_i
        for layer in self._layers:
            z = np.dot(input_layer, layer.weights.T) + layer.bias if layer.use_bias else 0
            y = layer.activation(z)
        return y

    def __gradient(self, error):
        pass

    def __update_weight(self, error, x_i):
        self._layers[-1].weights = self._layers[-1].weights + self._learning_rate*np.dot(error.T, x_i)

    def add(self, layer):
        self._layers.append(layer)
    
    def evaluate(self):
        pass

    def fit(self, X=None, Y=None, epochs=1, verbose=False): 
        for step in range(epochs):
            for x_i, y_i in zip(X,Y):
                # reshape x_i, y_i
                x_i =  x_i.reshape(1, X.shape[1])
                y_i = y_i.reshape(1, Y.shape[1])
                # propagacao para camadas da frente
                y_pred = self.__forward(x_i)
                error = y_i - y_pred
                # atualiza peso da utiliza camada
                self.__update_weight(error, x_i)

    def get_learning_rate(self):
        return self._learning_rate

    def get_weights(self):
        return self._layers[-1].weights, self._layers[-1].bias

    def predict(self, X, verbose=False):
        y_pred = []
        for i in range(len(X)):
            y_pred.append(self.__forward(X[i]))
        return y_pred

if __name__=='__main__':
    # x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # y = np.array([[0, 0, 0, 1]]).T
    # D = x.shape[1]
    
    # model = NeuralNetwork()
    # model.add(Layer(units=1, activation=relu, input_dim=D))
    # model.fit(x, y, epochs=10, verbose=False)

    # w,b = model.get_weights()
    # y_pred = model.predict(x, verbose=False)

    # print('w:', w)
    # print('b:', b)
    # print('x:', x)
    # print('y_pred:', y_pred)

    x,y = g.data_1A1(1000,dtype='train')
    D = x.shape[1]
    model = NeuralNetwork()
    model.add(Layer(units=8, activation=degrau, input_dim=D))
    model.fit(x, y, epochs=100, verbose=False)
    
    z,l = g.data_1A1(1,dtype='validation')
    
    w,b = model.get_weights()
    y_pred = model.predict(z, verbose=False)

    print('w:', w)
    print('b:', b)
    print('x:', z)
    print('y_pred:', y_pred)
    print('y_real:', l)