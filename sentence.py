def text():
    max_len = 20
    glove_dim = 50
    datadict = load_mr()
    embeddings_index = load_glove(glove_dim)
    X_train, y_train, X_test, y_test, labels_index = prepare_text_data(datadict,
                                                            embeddings_index,
                                                            max_len)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    nb_classes = len(labels_index)
    nb_epoch = 10
    text_indices = [np.where(y_train == i)[0] for i in range(nb_classes)]
    tr_pairs, tr_y = create_pairs(X_train, text_indices, nb_classes)
    text_indices = [np.where(y_test == i)[0] for i in range(nb_classes)]
    te_pairs, te_y = create_pairs(X_test, text_indices, nb_classes)

    input_shape = (max_len, glove_dim)
    base_network = text_lstm_base(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    distance = Lambda(euclidean_distance,
                      output_shape=eucl_dist_output_shape)([processed_a, processed_b])
    model = Model(input=[input_a, input_b], output=distance)

    # cross fingers and train
    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms)
    model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
          validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
          batch_size=128,
          nb_epoch=nb_epoch)

    # compute final accuracy on training and test sets
    pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
    tr_acc = compute_accuracy(pred, tr_y)
    pred = model.predict([te_pairs[:, 0], te_pairs[:, 1]])
    te_acc = compute_accuracy(pred, te_y)

    print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

    # plot in 2 dimensions
    pred = base_network.predict(X_train) # TEzT / TRAIN
    x = pred[:,0]
    y = pred[:,1]
    colors = ['red', 'green', 'blue', 'purple', 'cyan',
              'yellow', 'magenta', 'black', 'beige', 'darkorange']
    fig = plt.figure(figsize=(8,8))
    plt.scatter(x, y, c=list(y_train),
                cmap=matplotlib.colors.ListedColormap(colors[0:nb_classes]))
    plt.show()

def create_pairs(x, digit_indices, nb_class):
    """
    Positive and negative pair creation.
    Alternates between positive and negative pairs.
    """
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(nb_class)]) - 1
    for d in range(nb_class):
        for i in range(n-1):
            z1, z2 = digit_indices[d][i], digit_indices[d][i+1]
            z3, z4 = digit_indices[d][i], digit_indices[d][i+2]
            pairs += [[x[z1], x[z2]]]
            pairs += [[x[z3], x[z4]]]
            inc = random.randrange(1, nb_class)
            dn = (d + inc) % nb_class
            dm = (d + random.randrange(1, nb_class)) % nb_class
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            z3, z4 = digit_indices[d][i], digit_indices[dm][i+1]
            pairs += [[x[z1], x[z2]]]
            pairs += [[x[z3], x[z4]]]
            #labels += [1, 0]
            labels += [1, 1, 0, 0]
    return np.array(pairs), np.array(labels)
