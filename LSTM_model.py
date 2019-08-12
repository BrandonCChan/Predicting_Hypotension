# -----------------------------------------------------------------------------------------------------------------------
# Flexible Code for training LSTM based models and testing different hyperparameters, also supports transfer learning.
# Purpose of model illustrated in this code is to predict hypotension using physiological time series data.
# Uses Sacred as a way of passing hyperparameter values to the model and logs the training metrics in a mongodb similar
# to how tensorboard does it.
#
# Brandon Chan 2019
# -----------------------------------------------------------------------------------------------------------------------
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

ex = Experiment('Experiment Name')
ex.observers.append(MongoObserver.create(url='localhost:27017', db_name='Experiment Results'))
ex.captured_out_filter = apply_backspaces_and_linefeeds

@ex.config
def my_config():
    ''' Default configuration of hyperparamters. can add more by using @ex.named_config and another function '''
    num_input = 60  # Please constrain between 0 and 60
    lag_len = 50 # Please constrain between 0 and 60

    batch_size = 64
    num_epoch = 1000

    learning_rate = 0.005

    lr_schedule = True # Toggle the use of cosine annealing. Needs to have a set upper and lower LR bound
    learning_rate_upper = 0.001
    learning_rate_lower = 0.00005

    lstm_units = 60
    lstm_units2 = 60

    dropout_lstm1 = 0.0
    dropout_lstm2 = 0.4
    dropout_layer = 0.4

    l2_lambda_lstm1 = 0.001
    l2_lambda_lstm2 = 0.001
    l2_lambda_dense = 0.001 # 0.00001 also common? (10e-4 or 10e-6)?

    loss_name = 'binary_crossentropy'

    name_features = ['AR', 'HR', 'SPO2-%']

    model_name = 'model_lag' + str(lag_len) + '_obs' + str(num_input)

    data_dir = '/mnt/data2/brandon/reprocessed_data_apr1/our_data/balanced_redux/normalized/' # directory path for internal cohort
    physionet_dir = '/mnt/data2/brandon/reprocessed_data_apr1/physionet_data/apr16/normalized/' # directory path for external cohort
    checkpoint_dir = '/mnt/data2/brandon/ScientificReports/'

    stop_limit = 500
    run_ID = 0

    print_cfm = True # Toggle printing a confusion matrix of the model evaluation

    log_test_score = True # Toggle the evaluation of the model on the test set

    '''
    Transfer Learning Specific Hyperparameters
    - Transfer learning will only happen if num_transfer_learn_samples is > 0 and transfer_learn is a valid filepath 
    '''
    num_transfer_learn_samples = 0
    layer_freeze = "" # Defining which layers should be frozen, if any. For example, passing "23" would freeze layers 2 and 3
    # Define filepath for model to be transfer learned
    transfer_learn = "/Docuemnts/trained_models/model.hdf5"
    model_name = 'TL_tuned_model_lag' + str(lag_len) + '_obs' + str(num_input) + '_tune' + str(num_transfer_learn_samples)

@ex.capture
def my_metrics(_run, logs):
    ''' Set up listener for metrics to log. '''
    _run.log_scalar("loss", float(logs.get('loss')))
    _run.log_scalar("val_loss", float(logs.get('val_loss')))
    _run.log_scalar("accuracy", float(logs.get('acc')))
    _run.log_scalar("val_accuracy", float(logs.get('val_acc')))
    #_run.log_scalar("auc_roc", float(logs.get('auc_roc')))
    #_run.log_scalar("val_auc_roc", float(logs.get('val_auc_roc')))

@ex.automain
def my_main(num_input, lag_len, batch_size, name_features, num_epoch,
            learning_rate, learning_rate_upper, learning_rate_lower,
            lstm_units, lstm_units2, dropout_lstm1, dropout_lstm2, dropout_layer,
            l2_lambda_lstm1, l2_lambda_lstm2, l2_lambda_dense, loss_name, model_name, run_ID,
            data_dir, physionet_dir, checkpoint_dir, stop_limit, log_test_score,
            transfer_learn, num_transfer_learn_samples, layer_freeze, print_cfm):
    ''' takes parameters specified in the config and uses them to train a model. '''
    #------------------------------------------------------------------------------------------------------------------
    import tensorflow as tf
    from keras.models import Sequential, load_model, Model
    from keras.layers import Input, Dense, Activation, LSTM, Dropout, GRU, BatchNormalization #, LeakyReLU
    from keras import backend as backend
    from keras import regularizers
    from keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN, Callback #,ReduceLROnPlateau
    import keras.optimizers as kerasOpt
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn import metrics
    import numpy as np
    import pickle
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # ------------------------------------------------------------------------------------------------------------------
    # Helper functions
    # TODO:// Move to a separate file to make code a bit more streamline/clean. Note: Sacred didnt like this before
    class SGDRScheduler(Callback):
        '''Cosine annealing learning rate scheduler with periodic restarts.
        # Usage
            ```python
                schedule = SGDRScheduler(min_lr=1e-5,
                                         max_lr=1e-2,
                                         steps_per_epoch=np.ceil(epoch_size/batch_size),
                                         lr_decay=0.9,
                                         cycle_length=5,
                                         mult_factor=1.5)
                model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
            ```
        # Arguments
            min_lr: The lower bound of the learning rate range for the experiment.
            max_lr: The upper bound of the learning rate range for the experiment.
            steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
            lr_decay: Reduce the max_lr after the completion of each cycle.
                      Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
            cycle_length: Initial number of epochs in a cycle.
            mult_factor: Scale epochs_to_restart after each full cycle completion.
        # References
            Blog post: jeremyjordan.me/nn-learning-rate
            Original paper: http://arxiv.org/abs/1608.03983
        '''

        def __init__(self,
                     min_lr,
                     max_lr,
                     steps_per_epoch,
                     lr_decay=1.0,
                     cycle_length=10,
                     mult_factor=2.0):

            self.min_lr = min_lr
            self.max_lr = max_lr
            self.lr_decay = lr_decay

            self.batch_since_restart = 0
            self.next_restart = cycle_length

            self.steps_per_epoch = steps_per_epoch

            self.cycle_length = cycle_length
            self.mult_factor = mult_factor

            self.history = {}

        def clr(self):
            '''Calculate the learning rate.'''
            fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
            return lr

        def on_train_begin(self, logs={}):
            '''Initialize the learning rate to the minimum value at the start of training.'''
            logs = logs or {}
            backend.set_value(self.model.optimizer.lr, self.max_lr)

        def on_batch_end(self, batch, logs={}):
            '''Record previous batch statistics and update the learning rate.'''
            logs = logs or {}
            self.history.setdefault('lr', []).append(backend.get_value(self.model.optimizer.lr))
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

            self.batch_since_restart += 1
            backend.set_value(self.model.optimizer.lr, self.clr())

        def on_epoch_end(self, epoch, logs={}):
            '''Check for end of current cycle, apply restarts when necessary.'''
            if epoch + 1 == self.next_restart:
                self.batch_since_restart = 0
                self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
                self.next_restart += self.cycle_length
                self.max_lr *= self.lr_decay
                self.best_weights = self.model.get_weights()

        def on_train_end(self, logs={}):
            '''Set weights to the values from the end of the most recent cycle for best performance.'''
            self.model.set_weights(self.best_weights)

    class DecayLr(Callback):
        def __init__(self, n_epoch, decay):
            super(DecayLr, self).__init__()
            self.n_epoch = num_epoch
            self.decay = decay

        def on_epoch_begin(self, epoch, logs={}):
            old_lr = backend.get_value(self.model.optimizer.lr)
            if epoch > 1 and epoch % self.n_epoch == 0:
                new_lr = self.decay * old_lr
                backend.set_value(self.model.optimizer.lr, new_lr)
            else:
                backend.set_value(self.model.optimizer.lr, old_lr)

    def auc_roc(y_true, y_pred):
        return tf.py_func(metrics.roc_auc_score, (y_true, y_pred), tf.double)

    def precision(y_true, y_pred):
        true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + backend.epsilon())
        return precision

    def recall(y_true, y_pred):  # Also known as sensitivity
        true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + backend.epsilon())
        return recall

    def specificity(y_true, y_pred):
        neg_y_true = 1 - y_true
        neg_y_pred = 1 - y_pred
        fp = backend.sum(neg_y_true * y_pred)
        tn = backend.sum(neg_y_true * neg_y_pred)
        specificity = tn / (tn + fp + backend.epsilon())
        return specificity

    # ------------------------------------------------------------------------------------------------------------------
    # For logging metrics with sacred
    class LogMetrics(Callback):
        def on_epoch_end(self, _, logs={}):
            my_metrics(logs=logs)

    # ------------------------------------------------------------------------------------------------------------------
    # LOAD IN DATA
    if os.path.isfile(transfer_learn) and num_transfer_learn_samples > 0:
        print('--------------------------')
        print("Reading Data for Transfer Learning from:", physionet_dir)
        print('--------------------------')
        with open(physionet_dir + "validate_X_obs" + str(num_input) + "_lag" + str(lag_len) + ".npy" , "rb") as f:
            validation_X = np.load(f)
        with open(physionet_dir + "validate_y_obs" + str(num_input) + "_lag" + str(lag_len) + ".npy" , "rb") as f:
            validation_y = np.load(f)
        with open(physionet_dir + "test_X_obs" + str(num_input) + "_lag" + str(lag_len) + ".npy", "rb") as f:
            test_X = np.load(f)
        with open(physionet_dir + "test_y_obs" + str(num_input) + "_lag" + str(lag_len) + ".npy", "rb") as f:
            test_y = np.load(f)
        with open(physionet_dir + "tune_X_obs" + str(num_input) + "_lag" + str(lag_len) + ".npy" , "rb") as f:
            train_X = np.load(f)
        with open(physionet_dir + "tune_y_obs" + str(num_input) + "_lag" + str(lag_len) + ".npy" , "rb") as f:
            train_y = np.load(f)

        if num_transfer_learn_samples <= 50:
            train_X, _, train_y, _ = train_test_split(train_X, train_y, train_size=int(num_transfer_learn_samples), stratify=train_y)
        unique, counts = np.unique(train_y, return_counts=True)
        print(np.asarray((unique,counts)).T)

    else:
        print('--------------------------')
        print("Reading Data from:", data_dir)
        print('--------------------------')
        with open(data_dir + "train_X_obs" + str(num_input) + "_lag" + str(lag_len) + ".npy", "rb") as f:
            train_X = np.load(f)
        with open(data_dir + "train_y_obs" + str(num_input) + "_lag" + str(lag_len) + ".npy", "rb") as f:
            train_y = np.load(f)
        with open(physionet_dir + "test_X_obs" + str(num_input) + "_lag" + str(lag_len) + ".npy", "rb") as f:
            test_X = np.load(f)
        with open(physionet_dir + "test_y_obs" + str(num_input) + "_lag" + str(lag_len) + ".npy", "rb") as f:
            test_y = np.load(f)
        with open(data_dir + "validate_X_obs" + str(num_input) + "_lag" + str(lag_len) + ".npy", "rb") as f:
            validation_X = np.load(f)
        with open(data_dir + "validate_y_obs" + str(num_input) + "_lag" + str(lag_len) + ".npy", "rb") as f:
            validation_y = np.load(f)

    # ------------------------------------------------------------------------------------------------------------------
    num_features = len(name_features)
    features = []
    if 'AR' in name_features:
        features += [0]
    if 'HR' in name_features:
        features += [1]
    if 'SPO2-%' in name_features:
        features += [2]
    train_X = train_X[:, :, features]
    test_X = test_X[:, :, features]
    validation_X = validation_X[:, :, features]

    print('Shape training:', train_X.shape)
    print('Shape validate:', validation_X.shape)
    print('Shape test:', test_X.shape)

    checkpoint_filepath = checkpoint_dir + model_name + '_' + str(run_ID) + ".hdf5"
    callbacks = [TerminateOnNaN(),
                 EarlyStopping(monitor='val_loss',
                               patience=stop_limit,
                               mode='auto'),
                 LogMetrics(),
                 ModelCheckpoint(checkpoint_filepath,
                                 monitor='val_acc',
                                 verbose=0,
                                 save_best_only=True,
                                 mode='max')]

    # ------------------------------------------------------------------------------------------------------------------
    if os.path.isfile(transfer_learn):
        ''' Initialize and compile model for transfer learning'''
        print('-----------------------------')
        print('USING PRETRAINED WEIGHTS!...')
        print('-----------------------------')
        model = load_model(transfer_learn,
                           custom_objects={'auc_roc':auc_roc,'precision':precision,'recall':recall,'specificity':specificity})
        if layer_freeze:
            if "1" in layer_freeze:
                model.layers[0].trainable = False
            if "2" in layer_freeze:
                model.layers[1].trainable = False
            if "3" in layer_freeze:
                model.layers[2].trainable = False

        optimizer = kerasOpt.adam()
        model.compile(loss=loss_name,
                      optimizer=optimizer,
                      metrics=['acc'])
        schedule = SGDRScheduler(min_lr=0.000005,
                                 max_lr=0.0005,
                                 steps_per_epoch=np.ceil(train_y.shape[0] / batch_size),
                                 lr_decay=0.9,
                                 cycle_length=5,
                                 mult_factor=1.5)
        callbacks.append(schedule)
    else:
        ''' Define and compile model using hyperparamerters from config '''
        main_input = Input(shape=(num_input, num_features), name='main_input')
        x = LSTM(lstm_units,
                kernel_regularizer=regularizers.l2(l2_lambda_lstm1),
                recurrent_regularizer=regularizers.l2(l2_lambda_lstm1),
                return_sequences=True)(main_input)
        x = Dropout(dropout_lstm2)(x)
        x = LSTM(lstm_units2,
                kernel_regularizer=regularizers.l2(l2_lambda_lstm2),
                recurrent_regularizer=regularizers.l2(l2_lambda_lstm2))(x)
        x = Dropout(dropout_layer)(x)
        main_output = Dense(1, kernel_regularizer=regularizers.l2(l2_lambda_dense))(x)
        main_output = Activation('sigmoid', name='main_output')(main_output)
        model = Model(inputs=[main_input], outputs=[main_output])
        optimizer = kerasOpt.adam()
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy',
                      metrics=['acc', auc_roc])

        schedule = SGDRScheduler(min_lr=0.00005,
                                 max_lr=0.001,
                                 steps_per_epoch=np.ceil(train_y.shape[0] / batch_size),
                                 lr_decay=0.9,
                                 cycle_length=5,
                                 mult_factor=1.5)
        callbacks.append(schedule)

    print(model.summary())

    print(validation_y)
    model.fit(train_X, train_y,
              epochs=num_epoch,
              shuffle=True,
              batch_size=batch_size,
              callbacks=callbacks,
              verbose=2,
              validation_data=(validation_X, validation_y))

    print("Model done training!")

    def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
        """ Helper function to print confusion matrixes """
        columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
        empty_cell = " " * columnwidth
        # Print header
        print("    " + empty_cell, end=" ")
        for label in labels:
            print("%{0}s".format(columnwidth) % label, end=" ")
        print()
        # Print rows
        for i, label1 in enumerate(labels):
            print("    %{0}s".format(columnwidth) % label1, end=" ")
            for j in range(len(labels)):
                cell = "%{0}.1f".format(columnwidth) % cm[i, j]
                if hide_zeroes:
                    cell = cell if float(cm[i, j]) != 0 else empty_cell
                if hide_diagonal:
                    cell = cell if i != j else empty_cell
                if hide_threshold:
                    cell = cell if cm[i, j] > hide_threshold else empty_cell
                print(cell, end=" ")
            print()

    checkpoint_model = load_model(checkpoint_filepath, custom_objects={'auc_roc': auc_roc})

    '''
    Evaluate trained model on the validation set and if defined, also the test set. 
    '''
    validation_predictions = checkpoint_model.predict(validation_X)
    validation_predictions_binary = (np.array(validation_predictions) > 0.5) * 1
    TN, FP, FN, TP = confusion_matrix(validation_y, validation_predictions_binary).ravel()
    fpr, tpr, thresholds = metrics.roc_curve(validation_y, validation_predictions, pos_label=1)
    auc_score = metrics.auc(fpr, tpr)
    accuracy = accuracy_score(y_pred=validation_predictions_binary, y_true=validation_y)
    sensitivity_score = TP / (TP + FN)
    specificity_score = TN / (TN + FP)
    checkpoint_score_val = [accuracy, sensitivity_score, specificity_score, auc_score]
    print("Best Checkpoint Model Results:")
    print("[accuracy, sensitivity_score, specificity_score, auc_score]")
    print('Val accuracy:', checkpoint_score_val[0])
    print('Val sensitivity:', checkpoint_score_val[1])
    print('Val specificity:', checkpoint_score_val[2])
    print('Val auc:', checkpoint_score_val[3])

    if print_cfm == True:
        cm = confusion_matrix(validation_y, validation_predictions_binary)
        print_cm(cm,["Non-AHE","AHE"])

    if log_test_score is True:
        test_predictions = checkpoint_model.predict(test_X)
        test_predictions_binary = (np.array(test_predictions) > 0.5) * 1
        TN, FP, FN, TP = confusion_matrix(test_y, test_predictions_binary).ravel()
        fpr, tpr, thresholds = metrics.roc_curve(test_y, test_predictions, pos_label=1)
        auc_score = metrics.auc(fpr, tpr)
        accuracy = accuracy_score(y_pred=test_predictions_binary, y_true=test_y)
        sensitivity_score = TP / (TP + FN)
        specificity_score = TN / (TN + FP)
        checkpoint_score_test = [accuracy, sensitivity_score, specificity_score, auc_score]
        print("Checkpoint Model Results:")
        print("[accuracy, sensitivity_score, specificity_score, auc_score]")
        print('Test accuracy:', checkpoint_score_test[0])
        print('Test sensitivity:', checkpoint_score_test[1])
        print('Test specificity:', checkpoint_score_test[2])
        print('Test auc:', checkpoint_score_test[3])

        if print_cfm == True:
            cm = confusion_matrix(test_y, test_predictions_binary)
            print_cm(cm,["Non-AHE","AHE"])

    ex.add_artifact(checkpoint_filepath, name='best_model_weights')
    backend.clear_session()  # Clear session...

    if log_test_score is True:
        return checkpoint_score_test

    return checkpoint_score_val