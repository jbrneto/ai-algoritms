import keras
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Add, Activation, LayerNormalization 

def Block(n_filters, inputs, k_size=3, max_pooling=True, layer=0, decoding=False):
  name = 'Dec_'+ str(n_filters)+'_' if decoding else 'Enc_'+ str(n_filters)+'_'
  if styled:
      name = 'Sty_'+ str(n_filters)+'_'
  if layer > 0:
    name = name + '_' + str(layer) + '_'

  conv = inputs
  conv = Conv2D(n_filters, kernel_size=k_size, strides=1, padding='same', activation='relu', name=name+'Conv_1')(conv)
  conv = ReLU(name=name+'ReLU_1')(conv)
  conv = BatchNormalization(epsilon=1e-05, momentum=0.1, name=name+'BN_1')(conv)#, center=False, scale=False)(conv)
  #conv = Conv2D(n_filters, kernel_size=k_size, strides=1, padding='same', activation='relu', name=name+'Conv_2')(conv)
  #conv = ReLU()(conv)
  #conv = BatchNormalization(epsilon=1e-05, momentum=0.1)(conv)

  pool = None
  if max_pooling:
    pool = MaxPooling2D(pool_size=2, strides=2, padding='same', name=name+'Max')(conv)

  return conv, pool

def DecoderBlock(n_filters, inputs, skip_conenction, k_size=3, layer=0):
  name = 'Dec_'+str(n_filters)
  if layer > 0:
    name = name + '_' + str(layer) + '_'
  conv = Conv2DTranspose(n_filters, kernel_size=k_size-1, strides=2, padding='same', name=name+'Tran')(inputs)
  conv = tf.concat([conv, skip_conenction], axis=-1, name=name+'Concat')
  conv, _ = Block(n_filters, conv, k_size=k_size, max_pooling=False, layer=layer, decoding=True)
  return conv

def Unet(input_shape, styled=False):
  inputs = Input(input_shape, batch_size=32)

  # Encoding
  conv1, pool1 = Block(32, inputs)
  conv2, pool2 = Block(64, pool1)
  conv3, pool3 = Block(128, pool2)
  conv4, pool4 = Block(256, pool3)
  conv5, _ = Block(512, pool4, max_pooling=False)
  # Decoding
  uConv4 = DecoderBlock(254, conv5, conv4)
  uConv3 = DecoderBlock(128, uConv4, conv3)
  uConv2 = DecoderBlock(64, uConv3, conv2)
  uConv1 = DecoderBlock(32, uConv2, conv1)
  conv = Conv2D(1, activation='sigmoid', kernel_size=1, strides=1, padding='same', name="Top_Conv")(uConv1)

  model = Model(inputs=inputs, outputs=conv, name='UNet')
  return model

# Model Creation
input_shape = (256, 256, 3)    
model = Unet(input_shape)

optim = tf.keras.optimizers.Adam()
loss_fn = keras.losses.SparseCategoricalCrossentropy()
metrics = [tf.keras.metrics.AUC()]

model.compile(optimizer=optim, loss=loss_fn, metrics=metrics)