import numpy as np



def convert_to_tensor(str_file, isLabel):
    raw_data = [data.split(',') for data in str_file]
    dataset = []
    for per_data in raw_data:
        if isLabel:
            dataset = [int(data) for data in per_data]
        else:
            num_data = [float(data) for data in per_data]
            dataset.append(num_data)
    tensor = np.array(dataset)
        
    return tensor


def shuffle(X, Y):
    randomizer = np.arange(len(X))
    np.random.shuffle(randomizer)
    return (X[randomizer], Y[randomizer])


def imgs_filter(features_dataset,labels_dataset, target_label):
    tumor_idx, tumor_img = [], []
    for idx in range(len(labels_dataset)):
        if labels_dataset[idx] == target_label :
            tumor_idx.append(idx)
            tumor_img.append(features_dataset[idx])
        
    tumor_tensor = np.array(tumor_img)

    return tumor_idx, tumor_tensor