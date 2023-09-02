def get_dt():

    content_train_x = open(r"../English-german/train-x", "r")

    content_train_y = open(r"../English-german/train-y", "r")

    content_train_x = content_train_x.read().split("\n")
    content_train_y = content_train_y.read().split("\n")

    return content_train_x, content_train_y