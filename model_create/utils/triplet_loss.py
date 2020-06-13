import tensorflow as tf

from keras import backend as K


def _pair_dist(embeddings, squared=False):
    dot_product = tf.matmul(embeddings, K.transpose(embeddings))
    square_norm = tf.linalg.diag_part(dot_product)

    distances = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
    distances = tf.maximum(distances, 0.0)

    if not squared:
        mask = tf.cast(tf.equal(distances, 0.0), float)
        distances = distances + mask * 1e-16
        distances = tf.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def _get_triplet_mask(labels):
    indices_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)
    i_not_equal_j = tf.expand_dims(indices_not_equal, 2)
    i_not_equal_k = tf.expand_dims(indices_not_equal, 1)
    j_not_equal_k = tf.expand_dims(indices_not_equal, 0)

    distinct_indices = tf.logical_and(tf.logical_and(i_not_equal_j, i_not_equal_k), j_not_equal_k)

    label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    i_equal_j = tf.expand_dims(label_equal, 2)
    i_equal_k = tf.expand_dims(label_equal, 1)

    valid_labels = tf.logical_and(i_equal_j, tf.logical_not(i_equal_k))
    mask = tf.logical_and(distinct_indices, valid_labels)

    return mask


def _batch_all_triplet_loss(labels, embeddings, margin):
    pairwise_dist = _pair_dist(embeddings)
    a_p_dist = tf.expand_dims(pairwise_dist, 2)
    a_n_dist = tf.expand_dims(pairwise_dist, 1)

    margin = 0.5
    loss = a_p_dist - a_n_dist + margin

    mask = _get_triplet_mask(labels)
    mask = tf.cast(mask, float)
    loss = tf.multiply(mask, loss)
    loss = tf.maximum(loss, 0.0)
    valid_triplets = tf.cast(tf.greater(loss, 1e-16), float)
    num_positive_triplets = tf.reduce_sum(valid_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    loss = tf.reduce_sum(loss) / (num_positive_triplets + 1e-16)

    return loss, fraction_positive_triplets


def loss_function_maker(batch_size, margin=0.6):
    def batch_all(labels, embeddings, margin):
        loss, _ = _batch_all_triplet_loss(labels, embeddings, margin)
        return loss

    loss = batch_all

    def loss_function(y_true, y_pred):
        y_true = tf.slice(y_true, [0, 0], [batch_size, 1])
        print(y_true)
        y_true = tf.reshape(y_true, (tf.shape(y_true)[0],))
        print(y_true)
        print(y_pred)
        return loss(y_true, y_pred, margin)

    return loss_function
