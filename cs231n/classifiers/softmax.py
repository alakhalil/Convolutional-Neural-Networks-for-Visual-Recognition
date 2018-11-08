import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  N = X.shape[0] #number of instances
  C = W.shape[1] #number of classes
  #D = W.shape[0]
  ##### calculationg total loss ##########
  
  for i in range(N):
      #row_probabilities_sum = 0.0 #sum of probabilites of given instance
      scores_i = np.dot(X[i],W)#calculates scores 
      scores_i -= np.max(scores_i) #for number stability in order not to divide to large numbers
      
      probability = lambda c: np.exp(scores_i[c])/np.sum(np.exp(scores_i))
      
      
      loss-=np.log(probability(y[i])) #loss function        
  
      for c in range(C):
          p_c = probability(c)
          dW[:,c]  += (p_c-(c==y[i]))*X[i]
  
  loss /= N  #averaging
  loss+=reg*np.sum(W**2) #adding regularization term
  dW /= N
  dW += reg*W
  
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def dscore(W,X,y,reg,b):
  #method that dscores
  loss = 0.0
  
  scores = np.dot(X,W) + b #calculate scores
  
  scores = scores-np.max(scores,axis=1,keepdims=True) #subtract maximamum
  e = np.exp(scores) #raise scores as power of exponential
  probabilities = e / np.sum(e,axis=1,keepdims=True)  # divide by sum to get probabilities <=1 and >=0
  regularization_term = reg* np.sum(W**2) 
  loss = (np.sum(-1*np.log(probabilities[range(X.shape[0]),y])) / X.shape[0] ) + regularization_term #loss function
  
  dscores = probabilities
  dscores[np.arange(X.shape[0]),y]-=1 #probabilities -1(i=y)
  return loss,dscores       

def softmax_loss_vectorized(W, X, y, reg,b=None):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  if b is None:
      b = np.zeros(W.shape[1])
  loss, dscores = dscore(W,X,y,reg,b)
  


  dW = np.zeros_like(W)
  dW = np.dot(X.T,dscores)  #dL/dW = dL/dscores * dscores/dW = X.T x dL/dscores 
  
  dW /= X.shape[0] #average
  
  dW+=reg*W #regularization
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

