def write_history(history, filename):
    print("Writing history to file: {}".format(filename))

    with open(filename, 'w') as file:
        file.write("epoch,loss,acc,val_loss,val_acc\n")

        loss = history.history['loss']
        acc = history.history['acc']
        val_loss = history.history['val_loss']
        val_acc = history.history['val_acc']

        for i in range(len(loss)):
            file.write("{},{},{},{},{}\n".format(i + 1, loss[i], acc[i], val_loss[i], val_acc[i]))


def save_weights(model, model_name):
    model.layers.pop(0)
    model.save_weights(model_name + '-no-top.h5')
