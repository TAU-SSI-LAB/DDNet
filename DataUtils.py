from DataTransforms import *
from GeneralUtils import *


def extract_images_list_from_data_path(data_sets, data_percentage=1,
                                       extension=["jpg", "jpeg", "tiff", "tif", "png", "mat"]):

    out_list = []
    for set in data_sets:
        images_list = []
        labels_list = []
        assert len(set) == 2, 'Each data set must be a tuple of image dir path and label dir path.'
        images_dir, labels_dir = set

        images_list.extend(os.path.join(images_dir, image_id) for image_id in os.listdir(images_dir) if
                            (image_id.split(".")[1] == np.array(extension)).any())
        labels_list.extend(os.path.join(labels_dir, label_id) for label_id in os.listdir(labels_dir) if
                            (label_id.split(".")[1] == np.array(extension)).any())

        combined_list = [(os.path.split(image)[-1], os.path.split(image)[-2], os.path.split(label)[-2]
        if label is not None else None) for image, label in zip(images_list, labels_list)]

        if data_percentage < 1:
            combined_list = random.sample(combined_list, int(data_percentage * len(combined_list)))
        out_list.append(combined_list)

    return out_list


def split_train_validation_images(images_dirs, train_val_ratio, data_percentage):

    train_list = []
    val_list = []
    train_dirs_sizes = []

    images_list = extract_images_list_from_data_path(images_dirs, data_percentage)

    for images_list_from_dir in images_list:
        n_total = images_list_from_dir.__len__()
        n_train = np.ceil(n_total * train_val_ratio).astype(int)
        sampled_images_list_from_dir = random.sample(images_list_from_dir, n_total)
        random.shuffle(sampled_images_list_from_dir)
        train_list += sampled_images_list_from_dir[:n_train]
        val_list += sampled_images_list_from_dir[n_train:]
        train_dirs_sizes.append(n_train)
    return train_list, val_list, train_dirs_sizes


def get_train_val_images_list(train_sets, validation_sets, train_val_ratio, data_percentage=1.0):

    if validation_sets is None:
        return split_train_validation_images(train_sets, train_val_ratio, data_percentage)
    else:

        train_lists, valid_lists = list(map(lambda p: extract_images_list_from_data_path(p, data_percentage),
                                            [train_sets, validation_sets]))
        train_dirs_sizes = [len(train_set) for train_set in train_lists]

    return lists_to_flat_list(train_lists), lists_to_flat_list(valid_lists), train_dirs_sizes


def get_test_images_list(test_sets, data_percentage=1.0):

    return lists_to_flat_list(extract_images_list_from_data_path(test_sets, data_percentage))


def worker_init_fn(worker_id, random_seed=42):
    np.random.seed(worker_id + random_seed)


def lists_to_flat_list(list_of_lists):
    """
    Transforms list of lists into one list.
    """

    assert isinstance(list_of_lists, list)
    for _list in list_of_lists:
        assert isinstance(_list, list)
    return [item for sublist in list_of_lists for item in sublist]
