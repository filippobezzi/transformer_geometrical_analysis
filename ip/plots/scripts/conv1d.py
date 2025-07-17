def conv1d(x, scope, nf, *, w_init_stdev=0.02):
    with tf.variable_scope(scope):
        *start, nx = shape_list(x)
        w = tf.get_variable(
            'w',
            [1, nx, nf],
            initializer=tf.random_normal_initializer(stddev=w_init_stdev)
        )
        b = tf.get_variable(
            'b',
            [nf],
            initializer=tf.constant_initializer(0)
        )
        c = tf.reshape(
            tf.matmul(
                tf.reshape(x, [-1, nx]),
                tf.reshape(w, [-1, nf])
            ) + b,
            start + [nf]
        )
        return c