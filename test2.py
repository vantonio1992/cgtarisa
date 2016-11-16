for i in range(1000):
    batch_xs, batch_ys = get_batch(train_data,train_batch)
    if i%100 == 0:
        train_error = error.eval(feed_dict = {x_image: batch_xs, risa_out: risa_sqrt, W_t: W_t_np})
        print("step %d, training error %g"%(i, train_error))
    train_step.run(feed_dict={x_image: batch_xs, risa_out: risa_sqrt, W_t: W_t_np, keep_prob: 0.5})