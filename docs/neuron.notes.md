# Neuron


## Neuron logic. For one neuron only

One neuron solves an equation: y = weight * input + bias
weight? bias? => min cost (Ytarget - Ynow)


1) take input    
2) add bias to input
3) apply activation function to 2) 
4) make prediction i.e weight * input
5) count cost function
6) change weight (change slope, so we need to know should be increase or decrease the prime slope)


outputReal     AF(outputTarget)
---

cost rises = weight falls
cost falls = weight rises
cost c = weight cost i.e.  weight + deltaW  where  deltaW = 0 as cost close to 0
so weight + cost * learning_delta


### Possible variables values

input [0,inf)
y [0,inf)

degrees of prime (0, 90) = degrees [0, 45] = w[0,1] +  degrees (45, 90) = w (1, inf)
slope from 0 to 
y [0,inf)
cost [0,inf]

------------
Q&A

1) Define relation between cost function and activation function. How result of the cost function changes weight?
2) how to normalize w random selection? taking in account that P to select w [0,1] should be = P to select w (1,inf]? 0 -> 45 - > 90 degrees
any random value close to 1 would be ok?
3) why if input is greater then output it takes less steps to find a result?