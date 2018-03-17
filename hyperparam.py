'''
January 2018 by Daisy Tsai.
daisy3607@gmail.com
https://github.com/daisy3607/brain-tumor-detection
'''

class Hyperparams:
    '''Hyperparameters'''
    # train data
    train_fname_X = 'data/MRI_all.txt'
    train_fname_Y = 'data/MRI_all_labels.txt'
    # test_data
    test_fname_X = 'data/MRI_test.npy'
    test_fname_X= 'data/MRI_test.npy'
    # log_data
    log_fname = 'log/training_record.npy'
    # model fname
    model_fname = 'tumor_recognizer'

    img_cols, img_rows= 64, 64

    # training
    split_rate = 0.1 # split training data & testing data
    batch_size = 32 
    num_epochs = 20
    lr = 0.0001 # learning rate
    
    # model
    num_classes = 2 # num of tumor types
    dropout_rate = 0.1
    earlystopping_patient =  3


