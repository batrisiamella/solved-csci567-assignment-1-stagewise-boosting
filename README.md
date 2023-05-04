Download Link: https://assignmentchef.com/product/solved-csci567-assignment-1-stagewise-boosting
<br>
<h1>1           Boosting</h1>

In this problem, you will develop an alternative way of forward stagewise boosting. The overall goal is to derive an algorithm for choosing the best weak learner <em>h<sub>t </sub></em>at each step such that it best approximates the gradient of the loss function with respect to the current prediction of labels. In particular, consider a binary classification task of predicting labels <em>y<sub>i </sub></em>∈ {+1<em>,</em>−1} for instances <strong>x</strong><em><sub>i </sub></em>∈ R<em><sup>d</sup></em>, for <em>i </em>= 1<em>,…,n</em>. We also have access to a set of weak learners denoted by H = {<em>h<sub>j</sub>,j </em>= 1<em>,…,M</em>}. In this framework, we first choose a loss function <em>L</em>(<em>y<sub>i</sub>,y</em>ˆ<em><sub>i</sub></em>) in terms of current labels and the true labels, e.g. least squares loss <em>L</em>(<em>y<sub>i</sub>,y</em>ˆ<em><sub>i</sub></em>) = (<em>y<sub>i </sub></em>−<em>y</em>ˆ<em><sub>i</sub></em>)<sup>2</sup>. Then we consider the gradient <em>g<sub>i </sub></em>of the cost function <em>L</em>(<em>y<sub>i</sub>,y</em>ˆ<em><sub>i</sub></em>) with respect to the current predictions ˆ<em>y<sub>i </sub></em>on each instance, i.e.. We take the following steps for boosting:

<ul>

 <li><strong>Gradient Calculation </strong>In this step, we calculate the gradients.</li>

 <li><strong>Weak Learner Selection </strong>We then choose the next learner to be the one that can best predict these gradients, i.e. we choose</li>

</ul>

!

<em>h</em><sup>∗ </sup>= argmin

We can show that the optimal value of the step size <em>γ </em>can be computed in the closed form in this step, thus the selection rule for <em>h</em><sup>∗ </sup>can be derived independent of <em>γ</em>.

<ul>

 <li><strong>Step Size Selection </strong> We then select the step size <em>α</em><sup>∗ </sup>that minimizes the loss:</li>

</ul>

<em>n</em>

<em>α</em><sup>∗ </sup>= argmin<em><sub>α</sub></em>∈R<sup>X</sup><em>L</em>(<em>y<sub>i</sub>,y</em>ˆ<em><sub>i </sub></em>+ <em>αh</em><sup>∗</sup>(<strong>x</strong><em><sub>i</sub></em>))<em>.</em>

<em>i</em>=1

For the squared loss function, <em>α</em><sup>∗ </sup>should be computed analytically in terms of <em>y<sub>i</sub></em>, ˆ<em>y<sub>i</sub></em>, and <em>h</em><sup>∗</sup>. Finally, we perform the following updating step:

<em>y</em>ˆ<em><sub>i </sub></em>← <em>y</em>ˆ<em><sub>i </sub></em>+ <em>α</em><sup>∗</sup><em>h</em><sup>∗</sup>(<strong>x</strong><em><sub>i</sub></em>)<em>.</em>

In this question, you have to derive all the steps for squared loss function <em>L</em>(<em>y<sub>i</sub>,y</em>ˆ<em><sub>i</sub></em>) = (<em>y<sub>i </sub></em>− <em>y</em>ˆ<em><sub>i</sub></em>)<sup>2</sup>.

<h1>2           Neural Networks</h1>

<ul>

 <li> Show that a neural network with a single logistic output and with linear activation functions in the hidden layers (possibly with multiple hidden layers) is equivalent to the logistic regression.</li>

 <li>Consider the neural network in figure 1 with one hidden layer. Each hidden layer is defined as <em>z<sub>k </sub></em>= tanh(<sup>P3</sup><em><sub>i</sub></em><sub>=1 </sub><em>w<sub>ki</sub>x<sub>i</sub></em>) for <em>k </em>= 1<em>,…,</em>4 and the outputs are defined as <em>y<sub>j </sub></em>= <sup>P4</sup><em><sub>k</sub></em><sub>=1 </sub><em>v<sub>jk</sub>z<sub>k </sub></em>for <em>j </em>= 1<em>,</em> Suppose we choose the squared loss function for every pair, i.e. , where <em>y<sub>j </sub></em>and <em>y</em><sub>b</sub><em>j </em>represent the true outputs and our estimations, respectively. Write down the backpropagation updates for estimation of <em>w<sub>ki </sub></em>and <em>v</em><em>jk</em>.</li>

</ul>

1

Figure 1: A neural network with one hidden layer

<strong>Programming</strong>

<h1>Deep Learning</h1>

In this programming problem, you will be introduced to deep learning via hands on experimentation. We will explore the effects of different activation functions, training techniques, architectures and parameters in neural networks by training networks with different architectures and hyperparameters for a classification task.

For this homework, we highly recommend using the Google Cloud to run your code since training neural networks can take several tens of hours on personal laptops. You will need all the multi-core speedup you can get, to speed things up. We will only work with Python this time (no MATLAB), since all the deep learning libraries we need, are freely available only for Python.

There is an accompanying code file along with this homework titled hw_utils.py. It contains four functions which are all the functions you will need for the homework. You will not have to write any deep learning code by yourself for this homework, instead you will just call these helper functions with different parameter settings. Go over the file hw_utils.py and understand what each of the helper functions do.

<ul>

 <li><strong>Libraries</strong>: Launch a virtual machine on the Google Cloud (please use a 64-bit machine with Ubuntu 16.04 LTS and the maximum number of CPU cores you can get). Begin by updating the package list and then installing libraries

  <ul>

   <li><strong>Update package list</strong>: sudo apt-get update</li>

   <li><strong>Python Package Manager (pip)</strong>: sudo apt-get install python-pip</li>

   <li><strong>Numpy and Scipy</strong>: Standard numerical computation libraries in Python. Install with:</li>

  </ul></li>

</ul>

sudo apt-get install python-numpy python-scipy

<ul>

 <li><a href="http://deeplearning.net/software/theano/install_ubuntu.html">Theano</a><a href="http://deeplearning.net/software/theano/install_ubuntu.html">:</a> Analytical math engine. Install with:</li>

</ul>

sudo apt-get install python-dev python-nose g++ libopenblas-dev git sudo pip install Theano

<ul>

 <li><a href="https://keras.io/#installation">Keras</a><a href="https://keras.io/#installation">:</a> A popular Deep Learning library. Install with:</li>

</ul>

sudo pip install keras

<ul>

 <li><strong>Screen</strong>: For saving your session so that you can run code on the virtual machine even when you are logged out of it. Install with:</li>

</ul>

sudo apt-get install screen

Next, configure Keras to use <strong>Theano </strong>as its backend (by default, it uses <strong>TensorFlow</strong>). Open the <strong>Keras </strong>config file and change the backend field from tensorflow to theano. The Keras config file keras.json can be edited on the terminal with nano:

nano ~/.keras/keras.json

<ul>

 <li><strong>Useful information for homework</strong>: We will only use fully connected layers for this homework in all networks. We will refer to network architecture in the format: [<em>n</em><sub>1</sub><em>,n</em><sub>2</sub><em>,</em>·· <em>,n<sub>L</sub></em>] which defines a network having <em>L </em>layers, with <em>n</em><sub>1 </sub>being the input size, <em>n<sub>L </sub></em>being the output size, and the others being hidden layer sizes, e.g. the network in figure 1 has architecture:</li>

</ul>

[3<em>,</em>4<em>,</em>2].

Checkout the various activation functions for neural networks namely linear (<em>f</em>(<em>x</em>) = <em>x</em>), <a href="https://en.wikipedia.org/wiki/Sigmoid_function">sigmoid</a><a href="https://en.wikipedia.org/wiki/Sigmoid_function">,</a> <a href="https://en.wikipedia.org/wiki/Rectifier_(neural_networks)">ReLu</a> and <a href="https://en.wikipedia.org/wiki/Softmax_function">softmax</a><a href="https://en.wikipedia.org/wiki/Softmax_function">.</a> In this homework we will always use the softmax activation for the output layer, since we are dealing with a classification task and output of softmax layer can be treated as probabilities for multi-class classification.

Have a look at the last part of the homework (hints and tips section) before you start the homework to get some good tips for debugging and running your code fast.

A brief description of the functions in the helper file hw_utils.py is as follows:

<ul>

 <li>genmodel(): Returns a neural network model with the requested shape, activation function and L2-regularizer. You won’t need to call this method at all.</li>

 <li>loaddata(): Loads the dataset for this homework, shuffles it, generates labels, bifurcates the data and returns the training and test sets.</li>

 <li>normalize(): Normalizes the training and test set features.</li>

 <li>testmodels(): It takes the following parameters: your training and test data, a list of model architectures, activation function (hidden layers and last layer), list of regularization coefficients, number of epochs for stochastic gradient descent (SGD), batch size for SGD, learning rate for SGD, list of step size decays for SGD, list of SGD momentum parameters, boolean variable to turn nesterov momentum on/off, boolean variable to turn early stopping on/off and another boolean variable to turn the verbose flag on/off. The method generates a model of appropriate size and trains it on your training data. It prints out the test set accuracy on the console. In case of list of parameters, it trains networks for all possible combinations of those parameters and also reports the best configuration found (i.e. the configuration which gave the maximum test accuracy). This is the method that you will have to call a lot in your code.</li>

</ul>

Lastly, try running the experiments multiple times if needed, since neural networks are often subject to local minima and you might get suboptimal results in some cases.

<ul>

 <li><strong>Dataset and preprocessing</strong>: We will use the <a href="https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification">MiniBooNE particle identification dataset </a>from the UCI Machine Learning Repository. It has 130064 instances with 50 features each and each instance has to be classified as either ”signal” or ”background”.</li>

</ul>

Download the dataset and call loaddata() in your code to load and process it. The function loads the data, assigns labels to each instance, shuffles the dataset and randomly divides it into training (80%) and test (20%) sets. It also makes your training and test set labels categorical i.e. instead of a scalar ”0” or ”1”, each label becomes a two-dimensional tuple; the new label is (1,0) if the original label is ”0” and it is (0,1) if the original label is ”1”. The dimension of every feature is <em>d<sub>in </sub></em>= 50 and the dimension of output labels is <em>d<sub>out </sub></em>= 2. Next, normalize the features of both the sets by calling normalize() in your code.

<ul>

 <li><strong>Linear activations</strong>: (5 Points) First we will explore networks with linear activations. Train models of the following architectures: [<em>d<sub>in</sub></em>, <em>d<sub>out</sub></em>], [<em>d<sub>in</sub></em>, 50, <em>d<sub>out</sub></em>], [<em>d<sub>in</sub></em>, 50, 50, <em>d<sub>out</sub></em>], [<em>d<sub>in</sub></em>, 50, 50, 50, <em>d<sub>out</sub></em>] each having linear activations for all hidden layers and softmax activation for the last layer. Use 0.0 regularization parameter, set the number of epochs to 30, batch size to 1000, learning rate to 0.001, decay to 0.0, momentum to 0.0, Nesterov flag to False, and Early Stopping to False. Report the test set accuracies and comment on the pattern of test set accuracies obtained. Next, keeping the other parameters same, train on the following architectures: [<em>d<sub>in</sub></em>, 50, <em>d<sub>out</sub></em>], [<em>d<sub>in</sub></em>, 500, <em>d<sub>out</sub></em>], [<em>d<sub>in</sub></em>, 500, 300, <em>d<sub>out</sub></em>], [<em>d<sub>in</sub></em>, 800, 500, 300, <em>d<sub>out</sub></em>], [<em>d<sub>in</sub></em>, 800, 800, 500, 300, <em>d<sub>out</sub></em>]. Report the observations and explain the pattern of test set accuracies obtained. Also report the time taken to train these new set of architectures.</li>

 <li><strong>Sigmoid activation</strong>: (5 Points) Next let us try sigmoid activations. We will only explore the bigger architectures though. Train models of the following architectures: [<em>d<sub>in</sub></em>, 50, <em>d<sub>out</sub></em>],</li>

</ul>

[<em>d<sub>in</sub></em>, 500, <em>d<sub>out</sub></em>], [<em>d<sub>in</sub></em>, 500, 300, <em>d<sub>out</sub></em>], [<em>d<sub>in</sub></em>, 800, 500, 300, <em>d<sub>out</sub></em>], [<em>d<sub>in</sub></em>, 800, 800, 500, 300, <em>d<sub>out</sub></em>]; all hidden layers with sigmoids and output layer with softmax. Keep all other parameters the same as with linear activations. Report your test set accuracies and comment on the trend of accuracies obtained with changing model architectures. Also explain why this trend is different from that of linear activations. Report and compare the time taken to train these architectures with those for linear architectures.

<ul>

 <li><strong>ReLu activation</strong>: (5 Points) Repeat the above part with ReLu activations for the hidden layers (output layer = softmax). Keep all other parameters and architectures the same, except change the learning rate to 5 × 10<sup>−4</sup>. Report your observations and explain the trend again. Also explain why this trend is different from that of linear activations. Report and compare the time taken to train these architectures with those for linear and sigmoid architectures.</li>

 <li><strong>L2-Regularization</strong>: (5 Points) Next we will try to apply regularization to our network. For this part we will use a deep network with four layers: [<em>d<sub>in</sub></em>, 800, 500, 300, <em>d<sub>out</sub></em>]; all hidden activations ReLu and output activation softmax. Keeping all other parameters same as for</li>

</ul>

the previous part, train this network for the following set of L2-regularization parameters: [10<sup>−7</sup><em>,</em>5 × 10<sup>−7</sup><em>,</em>10<sup>−6</sup><em>,</em>5 × 10<sup>−6</sup><em>,</em>10<sup>−5</sup>]. Report your accuracies on the test set and explain the trend of observations. Report the best value of the regularization hyperparameter.

<ul>

 <li><strong>Early Stopping and L2-regularization</strong>: (5 Points) To prevent overfitting, we will next apply early stopping techniques. For early stopping, we reserve a portion of our data as a validation set and if the error starts increasing on it, we stop our training earlier than the provided number of iterations. We will use 10% of our training data as a validation set and stop if the error on the validation set goes up consecutively six times. Train the same architecture as the last part, with the same set of L2-regularization coefficients, but this time set the Early Stopping flag in the call to testmodels() as True. Again report your accuracies on the test set and explain the trend of observations. Report the best value of the regularization hyperparameter this time. Is it the same as with only L2-regularization? Did early stopping help?</li>

 <li><strong>SGD with weight decay</strong>: (5 Points) During gradient descent, it is often a good idea to start with a big value of the learning rate (<em>α</em>) and then reduce it as the number of iterations progress i.e.</li>

</ul>

In this part we will experiment with the decay factor <em>β</em>. Use the network [<em>d<sub>in</sub></em>, 800, 500, 300, <em>d<sub>out</sub></em>]; all hidden activations ReLu and output activation softmax. Use a regularization coefficient = 5 × 10−7, number of epochs = 100, batch size = 1000, learning rate = 10<sup>−5</sup>, and a list of decays: [10]. Use no momentum and no early stopping. Report your test set accuracies for the decay parameters and choose the best one based on your observations.

<ul>

 <li><strong>Momentum</strong>: (5 Points) Read about <a href="https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum">momentum</a> for Stochastic Gradient Descent. We will use a variant of basic momentum techniques called the Nesterov momentum. Train the same architecture as in the previous part (with ReLu hidden activations and softmax final activation) with the following parameters: regularization coefficient = 0.0, number of epochs = 50, batch size = 1000, learning rate = 10<sup>−5</sup>, decay = best value found in last part, Nesterov</li>

</ul>

= True, Early Stopping = False and a list of momentum coefficients = [0.99, 0.98, 0.95, 0.9, 0.85]. Find the best value for the momentum coefficients, which gives the maximum test set accuracy.

<ul>

 <li><strong>Combining the above</strong>: (10 Points) Now train the above architecture: [<em>d<sub>in</sub></em>, 800, 500, 300, <em>d<sub>out</sub></em>] (hidden activations: ReLu and output activation softmax) again, but this time we will use the optimal values of the parameters found in the previous parts. Concretely, use number of epochs = 100, batch size = 1000, learning rate = 10<sup>−5</sup>, Nesterov = True and Early Stopping = True. For regularization coefficient, decay and momentum coefficient use the best values that you found in the last few parts. Report your test set accuracy again. Is it better or worse than the accuracies you observed in the last few parts?</li>

 <li><strong>Grid search with cross-validation</strong>: (15 Points) This time we will do a full fledged search for the best architecture and parameter combinations. Train networks with architectures [<em>d<sub>in</sub></em>, 50, <em>d<sub>out</sub></em>], [<em>d<sub>in</sub></em>, 500, <em>d<sub>out</sub></em>], [<em>d<sub>in</sub></em>, 500, 300, <em>d<sub>out</sub></em>], [<em>d<sub>in</sub></em>, 800, 500, 300, <em>d<sub>out</sub></em>], [<em>d<sub>in</sub></em>, 800, 800, 500, 300, <em>d<sub>out</sub></em>]; hidden activations ReLu and final activation softmax. For each network use the following parameter values: number of epochs = 100, batch size = 1000, learning rate = 10<sup>−5</sup>, Nesterov = True, Early Stopping = True, Momentum coefficient = 0.99 (this is mostly independent of other values, so we can directly use it without including it in the hyperparameter search). For the other parameters search the full lists: for regularization coefficients = [10<sup>−7</sup><em>,</em>5 × 10<sup>−7</sup><em>,</em>10<sup>−6</sup><em>,</em>5 × 10<sup>−6</sup><em>,</em>10<sup>−5</sup>], and for decays = [10<sup>−5</sup><em>,</em>5 × 10<sup>−5</sup><em>,</em>10<sup>−4</sup>].</li>

</ul>

Report the best parameter values, architecture and the best test set accuracy obtained.

<strong>Hints and tips</strong>:

<ul>

 <li>You can use FTP clients like <a href="https://filezilla-project.org/">FileZilla</a> for transferring code and data to and fro from the virtual machine on the Google Cloud.</li>

 <li>Always use a screen session on the virtual machine to run your code. That way you can logout of the VM without having to terminate your code. A new screen session can be started on the console by: screen -S &lt;session_name&gt;. An existing attached screen can be detached by pressing ctrl+a followed by d. You can attach to a previously launched screen session by typing: screen -r &lt;session_name&gt;. Checkout the basic screen <a href="https://www.digitalocean.com/community/tutorials/how-to-install-and-use-screen-on-an-ubuntu-cloud-server">tutorial</a> for more commands.</li>

 <li>Don’t use the full dataset initially. Use a small sample of it to write and debug your code on your personal machine. Then transfer the code and the dataset to the virtual machine, and run the code with the full dataset on the cloud.</li>

 <li>While running your code, monitor your CPU usage using the top command on another instance of terminal. Make sure that if you asked for 24 CPUs, your usage for the python process is showing up to be around 2400% and not 100%. If top consistently shows 100%, then you need to setup your numpy, scipy and theano to use multiple cores. Theano (and Keras) by default make use of all cores, but numpy and scipy will not. Since raw numpy and scipy computations will only form a very small part of your program, you can ignore their parallelization for the most part, but if you so require you can google how to use multiple cores with numpy and set it up.</li>

 <li>Setting the verbose flag for fit() and evaluate() methods in Keras as 1 gives you detailed information while training. You can tweak this by passing verbose=1 in the testmodels() method in the py file.</li>

 <li>Remember to save your results to your local machine and turn off your machine after you have finished the homework to avoid getting charged unnecessarily.</li>

</ul>