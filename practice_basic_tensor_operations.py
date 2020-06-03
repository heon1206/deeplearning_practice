from __future__ import print_function
import tensorflow as tf

tf.enable_eager_execution()

#tensor constant
a = tf.constant(1)
b = tf.constant(2)
c = tf.constant(3)

#tensor operation
add = tf.add(a,b)
sub = tf.subtract(a,b)
div = tf.divide(a,b)
mul = tf.multiply(a,b)

#access tensor value
print("add = ",add.numpy())
print("sub = ", sub.numpy())
print("divide = ",div.numpy())
print("mul = ", mul.numpy())
#mean,sum

sum = tf.reduce_sum([[1,2,3],[4,5,6]])#21
mean = tf.reduce_mean([[1,2,3],[4,5,6]])#21/3

print("mean =", mean.numpy())
print("sum = ", sum.numpy())

#matrix multiplication
matrix_1 = [[1,2],[3,4]]
matrix_2 = [[5,6],[7,8]]

mul = tf.matmul(matrix_1,matrix_2)

#print
print("mul =", mul.numpy())