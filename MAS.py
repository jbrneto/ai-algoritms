import keras
import tensorflow as tf

# Source: https://github.com/ContinualAI/avalanche/blob/2d289fa146e167a77555ceba35aa48dc215b43e6/avalanche/training/plugins/mas.py#L12
def MAS_loss(model, loss, fim, prev_params, lambd): 
    def custom_loss(y_true, y_pred):
        normal_loss = loss(y_true,y_pred)

        regularization_value=tf.constant([0.])
        for layer in range(len(fim)):
            regularization_value += tf.reduce_sum(fim[layer] * tf.math.pow(tf.math.abs(prev_params[layer]-model.trainable_variables[layer]), tf.constant(2.0)))

        return normal_loss + (regularization_value * lambd)
    return custom_loss

def compute_MAS_matrix(model, loss, ds_train):
    avg_delta = None

    for x, y in tqdm(ds_train):
        y = tf.Variable(y, dtype='float32')
        with tf.GradientTape(persistent=True) as tape:
            logits = model(x, training=True)
            
            # Mean Norm of logits
            loss_value = tf.math.reduce_mean(tf.norm(tf.concat(logits,1), 2, 1))

        grads = tape.gradient(loss_value, model.trainable_weights)
        del tape

        # Zero non-existent gradients
        for g in range(len(grads)):
            if grads[g] is None:
                grads[g] = np.zeros(model.trainable_weights[g].shape)
        
        # Join grads
        if avg_delta is None:
            avg_delta = []
            for i in range(len(grads)):
                avg_delta.append(np.absolute(grads[i]))
        else:
            for i in range(len(grads)):
                avg_delta[i] = avg_delta[i] + np.absolute(grads[i])
    
    # Average Grads
    for i in range(len(avg_delta)):
        avg_delta[i] = avg_delta[i] / len(ds_train)
    
    return avg_delta

# Train Loop

model = build_model() # any tf model
loss = loss_fn() # loss function
tasks = [] # array of datasets [(t1_train, t1_val),(t2_train, t2_val)...]

optimizer = keras.optimizers.Adam()
fim = None
prev_params = None

for ds_train, ds_val in tasks:

    # Create regularized loss
    loss_fn = loss
    if fim is not None:
        loss_fn = MAS_loss(model, loss, fim, prev_params, lambd=100)
    model.compile(loss=loss_fn, optimizer=optimizer)
    
    # Train
    model.fit(
        ds_train,
        steps_per_epoch=len(ds_train), 
        epochs=10, 
        validation_data=ds_val, 
        validation_steps=len(ds_val),
    )    

    # Compute next iter values
    prev_params = [tf.identity(model.trainable_variables[layer]) for layer in range(len(model.trainable_variables))]
    new_fim = compute_MAS_matrix(model, loss, ds_val)

    # Join knowledge
    if fim is None:
        fim = new_fim.copy()
    else:
        for layer in range(len(fim)):
            fim[layer] = fim[layer] + new_fim[layer]
            #fim[layer] = (0.5 * fim[layer]) + (0.5 * new_fim[layer])