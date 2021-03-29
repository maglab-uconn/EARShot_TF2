from numpy import power,min,max,floor
import tensorflow as tf
from tensorflow.keras import Model, Sequential, optimizers, losses
from tensorflow.keras.layers import Dense, GRU, Input, Masking, LSTM
from tensorflow.keras.callbacks import LearningRateScheduler

tf.keras.backend.set_floatx('float64')

'''
Learning rate adjustment functions.
'''
def noam_decay_lr(warmup):
    '''
    Wrapper to define noam decay; the wrapper method allows us to make the
    lr update depend on additional parameters.

    The maximal learning rate under this scheme occurs at epoch = warmup, and
    will be equal to initial/warmup.
    '''
    def schedule(epoch, lr):
        # learning scheduler takes current epoch and lr as passes
        lrate = lr*power(warmup,-0.5)*min([(epoch+1)*power(warmup,-1.5),power(epoch+1,-0.5)])
        return lrate
    return LearningRateScheduler(schedule)


def step_decay_lr(initial_lr, drop_factor,drop_every):
    '''
    Wrapper that just drops the learning rate by a fixed factor (drop_factor) every drop_every
    epochs.
    '''
    def schedule(epoch):
        exp_fac = floor((1+epoch)/drop_every)
        lrate = initial_lr*power(drop_factor,exp_fac)
        return lrate

    return LearningRateScheduler(schedule)


def polynomial_decay_lr(max_epochs,poly_pow):
    '''
    Wrapper that drops the learning rate to zero over max_epochs epochs, with
    shape given by poly_pow (set poly_pow = 1 to get linear decay).
    '''
    def schedule(epoch, lr):
        decay = power((1 - (epoch/max_epochs)),poly_pow)
        lrate = lr*decay
        return lrate

    return LearningRateScheduler(schedule)


def constant_lr(initial):
    '''
    Wrapper that just clamps the learning rate at the initial value forever.
    '''
    def schedule(epoch):
        return initial

    return LearningRateScheduler(schedule)



class EARSHOT(Model):
    '''
    EARSHOT model sub-classing tf.keras.Model
    '''
    def __init__(self, output_len, model_parameters):
        '''
        output_len = length of target vector
        model_parameters = model hyper parameters pulled from parameters.py
        '''
        super(EARSHOT, self).__init__(name='earshot')
        self.model_parameters = model_parameters
        self.mask = Masking(mask_value=-9999, name="mask")

        if self.model_parameters.hidden['type'] == "LSTM":
            self.hidden = LSTM(self.model_parameters.hidden['size'],
                             return_sequences=True, stateful=False,
                             name="LSTM")
        elif self.model_parameters.hidden['type'] == "GRU":
            self.hidden = GRU(self.model_parameters.hidden['size'],
                           return_sequences=True, name="GRU")

        # loss function and output activation are coupled, this sets them both
        if self.model_parameters.train_loss == 'CE':
            #self.loss = losses.BinaryCrossentropy(from_logits=True)
            self.loss = 'binary_crossentropy'
            self.activation = tf.nn.sigmoid
        elif self.model_parameters.train_loss == 'MSE':
            #self.loss = losses.MeanSquaredError()
            self.loss = 'mean_squared_error'
            self.activation = tf.nn.tanh

        # set learning rate schedule
        if list(self.model_parameters.learning_schedule.keys())[0] == 'noam':
            self.lr_sched = noam_decay_lr(self.model_parameters.learning_schedule['noam']['warmup'])
            lr = self.model_parameters.learning_schedule['noam']['initial']
        elif list(self.model_parameters.learning_schedule.keys())[0] == 'constant':
            self.lr_sched = constant_lr(self.model_parameters.learning_schedule['constant']['rate'])
            lr = self.model_parameters.learning_schedule['constant']['rate']
        elif list(self.model_parameters.learning_schedule.keys())[0] == 'polynomial':
            self.lr_sched = polynomial_decay_lr(self.model_parameters.learning_schedule['polynomial']['max_epochs'],
                                                self.model_parameters.learning_schedule['polynomial']['poly_pow'])
            lr = self.model_parameters.learning_schedule['polynomial']['initial']
        elif list(self.model_parameters.learning_schedule.keys())[0] == 'step':
            self.lr_sched = step_decay_lr(self.model_parameters.learning_schedule['step']['initial'],
                                          self.model_parameters.learning_schedule['step']['drop_factor'],
                                          self.model_parameters.learning_schedule['step']['drop_every'])
            lr = self.model_parameters.learning_schedule['step']['initial']

        # optimizer
        if list(self.model_parameters.optimizer.keys())[0] == 'ADAM':
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, **self.model_parameters.optimizer['ADAM'])
        elif list(self.model_parameters.optimizer.keys())[0] == 'SGD':
            self.optimizer = tf.keras.optimizers.SGD(learning_rate=lr, **self.model_parameters.optimizer['SGD'])

        self.dense_output = Dense(output_len, activation=self.activation)


    def call(self, inputs):
        '''
        Input is provided at training time.
        '''
        x = self.mask(inputs)
        x = self.hidden(x)
        return self.dense_output(x)


    def model(self, input_shape):
        '''
        Function for model introspection
        '''
        x = Input(input_shape)
        return Model(inputs=[x], outputs=self.call(x))
