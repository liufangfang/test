import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#this data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)
#hyperparameters
lr =0.001
training_iters = 100000
batch_size =128
#display_step = 10

n_inputs = 28
n_steps = 28
n_hidden_units = 128
n_classes = 10

#Graph input
x=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y = tf.placeholder(tf.float32,[None,n_classes])
#Define weights
weights={
    #(28,128)
    'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
    #(128,10)
    'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))
}
biases={
    #(128,)
    'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
    #(10,)
    'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))
}

def RNN(X,weights,biases):
    #hidden layer for input to cell
    #X(128batch,28steps,28inputs)
    #==>(128*28,28inputs)
    X =tf.reshape(X,[-1,n_inputs])
    #X_in==>(128batch*28steps,128hidden)
    X_in =tf.matmul(X,weights['in'])+biases['in']
    #X_in == > (128batch ,28steps, 128hidden)
    X_in =tf.reshape(X_in,[-1,n_steps,n_hidden_units])
    #cell
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    else:
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    _init_state =lstm_cell.zero_state(batch_size,dtype=tf.float32)
    outputs,states=tf.nn.dynamic_rnn(lstm_cell,X_in,initial_state=_init_state,time_major=False)

    #hidden layer for output as the final results
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        outputs = tf.unpack(tf.transpose(outputs, [1, 0, 2]))  # states is the last outputs
    else:
        outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results =tf.matmul(states[1],weights['out']+biases['out'])
    #results = tf.matmul(outputs[-1], weights['out'] + biases['out'])
    return results

pred =RNN(x,weights,biases)
cost =tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
train_op =tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy =tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#init =tf.initialize_all_variables()
#init = tf.global_variables_initializer()
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step =0
    while step*batch_size <training_iters:
        batch_xs,batch_ys =mnist.train.next_batch(batch_size)
        batch_xs =batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run([train_op],feed_dict={
            x:batch_xs,
            y:batch_ys,
        })
        if step%20 ==0:
            print(sess.run(accuracy,feed_dict={
                x:batch_xs,
                y:batch_ys,
        }))
        step+=1