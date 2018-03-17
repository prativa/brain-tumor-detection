'''
March 2018 by Wenjing Cai.
daisy3607@gmail.com

'''

class Hyperparams:
    '''Hyperparameters'''
    # data
    features_filename = 'data/MRI_all.txt'
    labels_filename = 'data/MRI_all.txt'
    img_cols, img_rows= 64, 64

    # training
    split_rate = 0.1 # split training data & testing data
    batch_size = 32 
    num_epochs = 20
    lr = 0.0001 # learning rate
    
    # dir
    logdir = 'log_dir/' # log directory
    model_savedir = 'model_dir' # model saving directory
    
    # model
    num_classes = 2 # num of tumor types
    hidden_units = 512
    dropout_rate = 0.1
    earlystopping_patient =  3


