import matplotlib.pyplot as plt

def plot_acc_loss(acc, loss):
    plt.figure(figsize=(16,6))
    plt.subplot(1,2,1)
    plt.plot(acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.subplot(1,2,2)
    plt.plot(loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()