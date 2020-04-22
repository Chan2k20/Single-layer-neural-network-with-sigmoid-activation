import numpy as np
import sys
####Read train data####
f= open(sys.argv[1])
data = np.loadtxt(f)
train = data[:,1:]
trainlabels = data[:,0]

onearray = np.ones((train.shape[0], 1))
train = np.append(train, onearray, axis = 1)

##read test data##
f=open(sys.argv[2])
data = np.loadtxt(f)
test = data[:,1:]
testlabels = data[:,0]

rows = train.shape[0]
cols = train.shape[1]

hidden_nodes = int(sys.argv[3])

###initialize all weights####
w = np.random.rand(hidden_nodes)

#check this command
W = np.random.rand(hidden_nodes, cols)

epochs = 10000
eta = 0.001
prevobj = np.inf

######calculate objective######
hidden_layer = np.matmul(train , np.transpose(W))

sigmoid = lambda x: 1/(1+np.exp(-x))
hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
#print("hidden_layer = " , hidden_layer)
#print("hidden_year shape=", hidden_layer.shape)

output_layer = np.matmul(hidden_layer , np.transpose(w))
#print("output_layer =" , output_layer)

obj = np.sum(np.square(output_layer - trainlabels))
#print("obj=", obj)

###############Begin gradient descent######
#stop = 0.001
i=0
stop = 0
while(prevobj - obj > stop and i<epochs):
	prevobj = obj
	dellw = (np.dot(hidden_layer[0,:],w) - trainlabels[0])*hidden_layer[0,:]
	for j in range(1, rows):
		dellw += (np.dot(hidden_layer[j,:] , np.transpose(w)) - trainlabels[j]) * hidden_layer[j,:]
	#update w
	w = w -eta*dellw
	dellW=[]
	for i in range(hidden_nodes):
		dells=[]
		dells = np.sum(np.dot(hidden_layer[0,:], w) - trainlabels[0])*w[i] * (hidden_layer[0,i]) * (1-hidden_layer[0,i]) * train[0]
		for j in range(1, rows):
			dells += np.sum(np.dot(hidden_layer[j,:],w) - trainlabels[j])*w[i] *(hidden_layer[j,i]) * (1- hidden_layer[j,i]) *train[j]            
		dellW.append(dells)
		#Update W
	for i in range(hidden_nodes):
		dellW_new = eta*dellW[i]
		W[i]= W[i]- eta*dellW_new
		#Recalculate objective
	hidden_layer = np.matmul(train , np.transpose(W))
	#print("hidden_layer = " , hidden_layer)
	hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])	
	#print("output_layer = ", output_layer)
	output_layer = np.matmul(hidden_layer , np.transpose(w))  
	obj = np.sum(np.square(output_layer - trainlabels))
	#print("obj =" , obj)
	i = i + 1


###prediction##
def predict(test):
    onearray = np.ones((test.shape[0], 1))
    test = np.append(test, onearray, axis = 1)
    hidden_layer = np.matmul(test , np.transpose(W))
    sigmoid = lambda x: 1/(1+np.exp(-x))
    hidden_layer = np.array([sigmoid(xi) for xi in hidden_layer])
    output_layer = np.matmul(hidden_layer , np.transpose(w))  
    #print(output_layer)
    result = np.round(output_layer)    
    return result

print("Predictions : " ,predict(test))


        
        
        
        
        
        
