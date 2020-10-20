from GeneralUtils import *


def print_metrics(metrics, epoch_samples, phase, logger=None):

    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    if logger is None:
        print("{}: {}".format(phase, ", ".join(outputs)))
    else:
        logger.info("{}: {}".format(phase, ", ".join(outputs)))


def show_loss_statistics(epochs_losses, valid=True):

    assert isinstance(epochs_losses, dict)

    for loss in epochs_losses.keys():
        loss_name, loss_phase = loss.split('_')

        if loss_phase == 'val':
            continue
        if valid:
            val_loss_values = epochs_losses[loss_name + '_' + 'val']
        train_loss_values = np.array(epochs_losses[loss])

        plt.figure()
        plt.plot(train_loss_values)
        if valid:
            plt.plot(val_loss_values)
            plt.legend(('Train', 'Validation'))
        else:
            plt.legend('Train')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(loss_name)
        plt.grid(True)
        plt.show()

    return

