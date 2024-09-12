import keras
import tensorflow as tf

def EWC_loss(model, loss, fim, prev_params, lambd):
    def custom_loss(y_true, y_pred):
        normal_loss = loss(y_true, y_pred)

        regularization = tf.constant([0.])
        for layer in range(len(fim)):
            regularization += tf.reduce_sum(fim[layer]*(prev_params[layer]-model.trainable_variables[layer])**2)

        return normal_loss + (regularization * lambd)
    
    return custom_loss

def compute_fisher_matrix(model, task_set, batch_size):
  # Build fisher matrixes dictionary: at each key it will store the Fisher matrix for a particular layer
  fisher_matrixes = {n: tf.zeros_like(p.value()) for n, p in enumerate(model.trainable_variables)}
    
  #for i, (imgs, labels) in enumerate(task_set.take(batch_size)):
  for imgs, labels in task_set:

    # Initialize gradients storage
    with tf.GradientTape() as tape:
      #preds = model(imgs)[task_id]
      preds = model(imgs)

      # Log of the predictions
      ll= tf.math.log(preds)

    # Log_likelihood grads
    ll_grads  = tape.gradient(ll, model.trainable_variables)
      
    # Compute Fisher matrix at each layer
    for i, gradients in enumerate(ll_grads):
        if gradients != None:
            fisher_matrixes[i] += tf.math.reduce_mean(gradients ** 2, axis=0) / batch_size

  return fisher_matrixes

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
        loss_fn = EWC_loss(model, loss, fim, prev_params, lambd=100)
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
    new_fim = compute_fisher_matrix(model, ds_val, batch_size=32)

    # Join knowledge
    if fim is None:
        fim = new_fim.copy()
    else:
        for layer in range(len(fim)):
            fim[layer] = fim[layer] + new_fim[layer]