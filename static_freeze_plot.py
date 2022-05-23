import matplotlib.pyplot as plt
import datetime
import os




def plot_diff(base_value, next_value, figure_idx, metric):
    plt.figure(figure_idx)
    plt.title(
        f'Gradually Freezing on LeNet-5 with CIFAR10 dataset metric: {metric}')
    plt.ylabel(metric)  # y label
    plt.xlabel("Epochs")  # x label
    plt.plot(base_value,
             label='Base (Current state)',
             marker="o",
             linestyle="-")
    plt.plot(next_value,
             label='Next (Freeze 1 more)',
             marker="o",
             linestyle="-")
    plt.legend()


baseline = [0.5056, 0.6122, 0.6544, 0.7101000000000001, 0.7386, 0.6662, 0.7535, 0.745, 0.7515999999999999, 0.7895, 0.7362000000000001, 0.7325, 0.7828, 0.7673000000000001, 0.7508, 0.7178, 0.7525, 0.7314, 0.7291, 0.7623000000000001]
e10_d5 = [0.4598, 0.5602, 0.6692, 0.7243, 0.7412000000000001, 0.7548999999999999, 0.7579, 0.77, 0.7468, 0.7431, 0.7505, 0.785, 0.7391, 0.784, 0.7709, 0.807, 0.7609, 0.7875, 0.7655, 0.7703]
e10_d10 = [0.41340000000000005, 0.5936, 0.6278, 0.7212000000000001, 0.7134999999999999, 0.727, 0.7448999999999999, 0.7191, 0.7659, 0.7673000000000001, 0.7806000000000001, 0.8104, 0.8177, 0.8240999999999999, 0.7844, 0.8029000000000001, 0.7936, 0.7514, 0.7584000000000001, 0.6805]
e10_d15 = [0.5466, 0.6159, 0.6665000000000001, 0.7212999999999999, 0.7384999999999999, 0.7197, 0.6999, 0.7503, 0.7633, 0.7746999999999999, 0.7272, 0.8198000000000001, 0.8145, 0.804, 0.8197, 0.8167, 0.8043, 0.7856000000000001, 0.7609, 0.6979000000000001]

e5_d5 = [0.4729, 0.6212, 0.6431, 0.7289, 0.7101000000000001, 0.7451000000000001, 0.7593000000000001, 0.7648999999999999, 0.7719, 0.7720999999999999, 0.7505, 0.772, 0.7322, 0.7365, 0.7637, 0.7887000000000001, 0.7684000000000001, 0.7218000000000001, 0.7281, 0.5567]
e5_d10 = [0.5022, 0.5927, 0.6345000000000001, 0.6453, 0.7349, 0.7176, 0.7927, 0.7795000000000001, 0.7895, 0.7796, 0.7861, 0.8017, 0.7816, 0.7774, 0.762, 0.7518, 0.5722, 0.684, 0.6414, 0.254]
e5_d15 = [0.4819, 0.6568, 0.6954, 0.6951, 0.7311, 0.7465999999999999, 0.8112999999999999, 0.8025, 0.7909, 0.8114, 0.8001999999999999, 0.7996, 0.7968000000000001, 0.8047, 0.78, 0.7544, 0.6683, 0.5714, 0.10339999999999999, 0.1]

def offline_plot(idx, title, y_title):
    plt.figure(idx)
    plt.title(title)
    plt.ylabel(y_title)  # y label
    plt.xlabel("Epochs")  # x label

    plt.plot(baseline,
                label='Baseline: No freeze',
                marker="o",
                linestyle="-")
    plt.plot(e5_d5, label='Freeze degree=5',
                marker="o", linestyle="--")

    plt.plot(e5_d10, label='Freeze degree=10',
                marker="o", linestyle="--")
    plt.plot(e5_d15, label='Freeze degree=15',
                marker="o", linestyle="--")
    plt.legend()



def save_figure(title):    
    now = datetime.datetime.now()
    dt_string = now.strftime("%m-%d-%Y_%H:%M:%S")

    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'results/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    image_filename = title + "_" + dt_string + ".png"
    plt.savefig(results_dir+image_filename)
    # plt.savefig(f'{title}_{dt_string}.png')


def show():
    plt.show()


if __name__ == '__main__':
    # offline_plot(1, "Gradually Freezing w/ Model Switching Accuracy", "Accuracy")
    offline_plot(1, "MobileNetV2 Static Freezing at epoch=5", "Accuracy")


    plt.show()
