import tensorflow as tf

# construction phase
x = tf.Variable(3, name='x')
y = tf.Variable(4, name='y')

f = x * x * y + y + 2

# execution phase
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()  # initializes all the variables / creates a graph
    result = f.eval()
    print(result)
