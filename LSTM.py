from Parameter import Parameter
from Helper import *

class LSTM:
    def __init__(self, learning_rate, max_iterations, time_step, input_shape, hidden_layers=10):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.time_step = time_step
        self.input_shape = input_shape
        self.hidden_layers = hidden_layers
        self.initialize_parameters()
        
    def initialize_parameters(self):
        # Parameters for Forgot gate
        self.W_F = Parameter('W_F', shape=(self.hidden_layers, self.hidden_layers+self.input_shape[0]), weight_sd=0.1, c=0.5)
        self.B_F = Parameter('B_F', shape=(self.hidden_layers, 1), weight_sd=0.1) 
        
        # Parameters for Input gate
        self.W_I = Parameter('W_I', shape=(self.hidden_layers, self.hidden_layers+self.input_shape[0]), weight_sd=0.1, c=0.5)
        self.B_I = Parameter('B_I', shape=(self.hidden_layers, 1), weight_sd=0.1)
        
        # Parameters for Ouput gate
        self.W_C = Parameter('W_C', shape=(self.hidden_layers, self.hidden_layers+self.input_shape[0]), weight_sd=0.1, c=0.5)
        self.B_C = Parameter('B_C', shape=(self.hidden_layers, 1), weight_sd=0.1)
        self.W_O = Parameter('W_O', shape=(self.hidden_layers, self.hidden_layers+self.input_shape[0]), weight_sd=0.1, c=0.5)
        self.B_O = Parameter('B_O', shape=(self.hidden_layers, 1), weight_sd=0.1)
        
        # Parameters for Output
        self.W_V = Parameter('W_V', shape=(self.input_shape[1], self.hidden_layers), weight_sd=0.1, c=0.5)
        self.B_V = Parameter('B_V', shape=(self.input_shape[1], 1), weight_sd=0.1)
        
    def forward(self, x):
        self.x = x
        h = [np.zeros((self.hidden_layers, 1)) for _ in range(self.time_step)]
        c = [np.zeros((self.hidden_layers, 1)) for _ in range(self.time_step)]
        f = [np.zeros((self.hidden_layers, 1)) for _ in range(self.time_step)]
        i = [np.zeros((self.hidden_layers, 1)) for _ in range(self.time_step)]
        c_bar = [np.zeros((self.hidden_layers, 1)) for _ in range(self.time_step)]
        o = [np.zeros((self.hidden_layers, 1)) for _ in range(self.time_step)]
        
        for step in range(1, self.time_step):
            z = np.vstack((h[step-1], x[step]))
            f[step] = sigmoid((self.W_F.value @ z) + self.B_F.value)
            i[step] = sigmoid((self.W_I.value @ z) + self.B_I.value)
            c_bar[step] = tanh((self.W_C.value @ z) + self.B_C.value)

            c[step] = (f[step] * c[step-1]) + (i[step] * c_bar[step])

            o[step] = sigmoid((self.W_O.value @ z) + self.B_O.value)

            h[step] = o[step] * tanh(c[step])

        self.f, self.i, self.c_bar, self.c, self.o, self.h = f, i , c_bar, c, o, h
        v = self.W_V.value @ self.h[-1] + self.B_V.value
        
        return v
        
    
    def backward(self, y, y_pred):
        h_diff = [np.zeros((self.hidden_layers, 1)) for _ in range(self.time_step + 1)]
        c_diff = [np.zeros((self.hidden_layers, 1)) for _ in range(self.time_step + 1)]
        
        delta_e = y - y_pred
        
        self.W_V.diff = delta_e * self.h[-1].T
        self.B_V.diff = delta_e
        
        for step in reversed(range(self.time_step)):
            h_diff[step] = self.W_V.value.T @ delta_e + h_diff[step+1]
            o_diff = tanh(self.c[step]) * h_diff[step] * sigmoid_diff(self.h[step])
            # print('o_diff\t{}\nh_diff\t{}'.format(o_diff.shape, h_diff[step].shape))
            c_diff[step] = self.o[step] * h_diff[step] * tanh_diff(self.c[step]) + c_diff[step+1]
            c_bar_diff = self.i[step] * c_diff[step] * tanh_diff(self.c_bar[step])
            i_diff = self.c_bar[step] * c_diff[step] * sigmoid_diff(self.i[step])
            f_diff = self.c[step-1] * c_diff[step] * sigmoid_diff(self.f[step])
            
            z = np.vstack((self.h[step-1], self.x[step]))
            
            self.W_F.diff += f_diff @ z.T
            self.B_F.diff += f_diff
            
            self.W_I.diff += i_diff @ z.T
            self.B_I.diff += i_diff
            self.W_O.diff += o_diff @ z.T
            self.B_O.diff += o_diff
            
            self.W_C.diff += c_diff[step] @ z.T
            self.B_C.diff += c_diff[step]
    
    def get_parameters(self):
        return [self.W_F, self.B_F, 
                self.W_I, self.B_I,
                self.W_C, self.B_C,
                self.W_O, self.B_O,
                self.W_V, self.B_V]
    
    def clear_gradients(self):
        for parameter in self.get_parameters():
            parameter.diff = np.zeros_like(parameter.value)
        
    def clip_gradients(self):
        for parameter in self.get_parameters():
            np.clip(parameter.diff, -1, 1, out=parameter.diff)
            
    def update_parameters(self):
        for parameter in self.get_parameters():
            if parameter.name == 'W_V':
                parameter.value += self.learning_rate * parameter.diff
            else:
                parameter.value += self.learning_rate * (parameter.diff/self.time_step)
            
    def fit(self, x, y):
        for epoch in range(self.max_iterations):
            loss = 0
            for i in range(len(x)):
                y_pred = self.forward(x[i]) 
                loss += ((y[i] - y_pred)**2)
                self.backward(y[i], y_pred)
                self.update_parameters()
                self.clear_gradients()
            print('epoch {}/{} - loss : {}'.format(epoch, self.max_iterations, loss/len(x)), end='\r')
            
    def predict(self, x):
        y_pred = []
        for i in range(len(x)):
            y_pred.append(self.forward(x[i]))
        return np.concatenate(y_pred)
