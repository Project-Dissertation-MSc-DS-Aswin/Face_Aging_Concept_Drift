import tensorflow as tf
import numpy as np

"""
Cosine Distance
"""
def cosine(emb_array1, emb_array2):
  return tf.matmul(
    tf.expand_dims(emb_array2, 0), tf.transpose(tf.expand_dims(emb_array1, 0))
  ) / (tf.norm(emb_array1, ord=2) / tf.norm(emb_array2, ord=2))

"""
Euclidean Distance
"""
def euclidean(emb_array1, emb_array2):
  return tf.norm(emb_array1 - emb_array2, ord=2)

"""
Euclidean Distance and Cosine distance
"""
def face_distance(i, x, embeddings):
  
  m = embeddings.shape[0]
  n = m**2
  
  if type(i) == int:
    i_2 = i
  else:
    i_2 = i.numpy().astype(int)
  emb_array1 = embeddings[i_2//m]
  emb_array2 = embeddings[i_2%m]
  cosine = tf.matmul(tf.expand_dims(emb_array2, 0), tf.transpose(tf.expand_dims(emb_array1, 0))) / (tf.norm(emb_array1, ord=2) / tf.norm(emb_array2, ord=2))
  norm = tf.norm(emb_array1 - emb_array2, ord=2)
  
  return tf.convert_to_tensor(np.array([norm, cosine], dtype=np.float64), dtype=tf.float64)

