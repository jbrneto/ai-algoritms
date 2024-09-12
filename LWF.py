import keras
import tensorflow as tf

# Train Loop

model = build_model() # any tf model
loss = loss_fn() # loss function
tasks = [] # array of datasets [(t1_train, t1_val),(t2_train, t2_val)...]

optimizer = keras.optimizers.Adam()
prev_w = None

first = True
for ds_train, ds_val in tasks:
    if first:
        model.fit(
            ds_train,
            steps_per_epoch=len(ds_train),
            epochs=10,
            validation_data=ds_val,
            validation_steps=len(ds_val)
        )
        first = False
        prev_w = model.get_weights()
        continue

    for e in range(10):
        for x, y in ds_train:
            y = tf.Variable(y, dtype='float32')

            # save current model
            curr_w = model.get_weights()
            # load old model
            model.set_weights(prev_w)
            model.compile(loss=loss, optimizer=optimizer)
            
            # predict new data with old model
            old_preds = model(x, training=False)

            # load new model back
            model.set_weights(curr_w)
            model.compile(loss=loss, optimizer=optimizer)
            
            with tf.GradientTape(persistent=True) as tape:
                preds = model(x, training=True)

                # convergence loss
                reg1 = loss(y, preds)
                # retention loss
                reg2 = loss(old_preds, preds)
                # generalization loss
                #reg3 = loss(y*old_preds, y*old_preds*preds)
               
                loss_value = reg1+reg2#+reg3
                
            grads = tape.gradient(loss_value, model.trainable_weights)
            quickopt.apply_gradients(zip(grads, model.trainable_variables))

    del curr_w
    prev_w = model.get_weights()