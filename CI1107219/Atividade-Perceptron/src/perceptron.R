library(ggplot2)

threshold <- function(y) {
  ifelse(y>0, 1, 0)
}

activation_func <- function(z, func) {
  switch(func, 
    'degrau' = ifelse(z>0, 1, 0),
    'sigmoid' = 1/(1+exp(-z)),
    'tanh' = (2/(1+exp(-2*z)))-1,
    'relu' = ifelse(z>0, z, 0))
}

forward <- function(w,b,x_i) {
  z = (x_i%*%t(w)) + b
  activation_func(z, 'degrau')[1,1]
}

train <- function(X, Y, epoch, learning_rate) {
  # inicializando pesos e bias
  w = matrix(runif(dim(X)[2])-0.5, nrow=1, ncol=dim(X)[2])
  b = 0
  # tamanho da entrada
  x_len = length(Y)
 
  for(step in 1:epoch) {
    for(i in 1:x_len) {
      y_pred = forward(w,b,X[i,])
      error = Y[i]-y_pred
      w = w + learning_rate*(X[i,]*error)
      b = b + learning_rate*error
    }
  }
  print(w)
  print(b)
}

x = matrix(c(0,0,0,1,1,0,1,1), nrow=4, ncol=2, byrow=TRUE)
y = c(0,0,0,1)

train(x, y, 1000, 0.03)