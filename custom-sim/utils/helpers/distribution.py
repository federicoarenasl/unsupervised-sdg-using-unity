import pandas as pd
import cv2
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import os
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
from tqdm import tqdm_notebook as tqdm

# Distribution helper functions
def get_epoch_dist(dist_path, epoch):
    dist_epoch_path = os.path.join(dist_path, f"epoch_{epoch}")
    return walk_json_folder(dist_epoch_path)


def get_epoch_imgs(imgs_path, epoch, max_samples):
    imgs_epoch_path = os.path.join(imgs_path, f"epoch_{epoch}")
    return walk_png_folder(imgs_epoch_path, max_samples)


def get_distributions(experiment_n, interv, parameter):
    graph_path = os.path.join(ROOT_PATH, f"experiment{experiment_n}", DIST_PATH)
    hpams = read_hyperaparameters(experiment_n)
    total_epochs = hpams["max_epochs"]
    freq = hpams["distprint_freq"]
    interval = interv

    dist_dict = {}
    for e in range(0, total_epochs + 1, freq * interval):
        dist_samples = get_epoch_dist(graph_path, e)
        if len(dist_samples) > 1:
            dist_dict[f"epoch_{e}"] = [
                dist_sample[parameter] for dist_sample in dist_samples
            ]

        else:
            print("Full results not ready yet")
            break

    return dist_dict


def get_tgt_distribution(experiment_n, parameter):
    tgt_path = os.path.join(ROOT_PATH, f"experiment{experiment_n}", TGT_PATH)
    tgt_dist_dict = walk_json_folder(tgt_path)
    return {"tgt_dist": [tgt[parameter] for tgt in tgt_dist_dict]}


def get_distdf(experiment_n, interval, parameter="yaw"):
    # Get target distributions
    dist_dict = get_distributions(
        experiment_n=experiment_n, interv=interval, parameter=parameter
    )
    tgt_dict = get_tgt_distribution(experiment_n=4, parameter=parameter)
    # Get dataframe
    dist_dict["tgt_dist"] = tgt_dict["tgt_dist"]

    return pd.DataFrame.from_dict(dist_dict)


def get_tgt_image(max_tgt_samples, experiment_n):
    tgt_path = os.path.join(ROOT_PATH, f"experiment{experiment_n}", TGT_PATH)
    tgt_image = walk_png_folder(tgt_path, max_tgt_samples)
    return tgt_image[0]


def get_gen_images(experiment_n, interv):
    imgs_path = os.path.join(ROOT_PATH, f"experiment{experiment_n}", DIST_PATH)
    hpams = read_hyperaparameters(experiment_n)
    total_epochs = hpams["max_epochs"]
    freq = hpams["distprint_freq"]
    max_samples = hpams["epoch_length"]
    max_tgt_samples = hpams["num_real_images"]
    interval = interv

    imgs_dict = {}
    for e in range(0, total_epochs + 1, freq * interval):
        dist_samples = get_epoch_imgs(imgs_path, e, max_samples)
        if len(dist_samples) == 1:
            imgs_dict[f"epoch_{e}"] = dist_samples[0]

        else:
            print("Full results not ready yet")
            break

    # Add target image
    imgs_dict["Target image"] = get_tgt_image(max_tgt_samples, experiment_n)
    return imgs_dict

def plot_distributions(experiment, interval, parameter="yaw", title="No title"):
    # Get target distribution
    tgt_dist_dict = get_tgt_distribution(experiment, parameter)
    # Get target distributions
    dist_dict = get_distributions(experiment, interval, parameter)
    # Creating figure
    sns.set_style("whitegrid")
    sns.set_style("ticks", {"xtick.major.size": 5, "ytick.major.size": 5})

    fig = plt.figure(figsize=(10, 3))
    colors = []
    # Plot generated distributions
    for epoch_n in dist_dict.keys():
        e = int(epoch_n.split("_")[-1])
        angles = dist_dict[epoch_n]
        if np.mean(angles) < 0.1:
            angles = list(np.array(angles))

        hist, edges = np.histogram(angles, bins=20)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        bin_widths = edges[1:] - edges[:-1]
        plt.bar(bin_centers, hist, width=bin_widths, label=epoch_n, joinstyle="bevel")
        plot = plt.plot(bin_centers, hist, linewidth=3)
        colors.append(plot[0].get_color())

    # Plot target distribution
    tgt_angles = tgt_dist_dict["tgt_dist"]
    # tgt_angles = list(np.array(tgt_angles)+ np.random.normal(0,0.1,np.array(tgt_angles).shape))

    hist, edges = np.histogram(tgt_angles, bins=20)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_widths = edges[1:] - edges[:-1]

    plt.bar(
        bin_centers,
        hist,
        width=bin_widths,
        alpha=0.7,
        label="Target dist.",
        joinstyle="bevel",
    )
    plot = plt.plot(bin_centers, hist, linewidth=3, color="salmon", alpha=0.7)
    colors.append(plot[0].get_color())
    # Finalize plot
    plt.title(title)
    plt.legend(loc="upper right")
    plt.xlabel(f"{parameter}")
    plt.ylabel("frequency")
    plt.show()


def plot_distribution_grid(
    experiment,
    rows=2,
    columns=2,
    size=(15, 8),
    interval=2,
    parameter="yaw",
    title="No title",
    i=1,
):
    # Get target distribution
    tgt_dist_dict = get_tgt_distribution(experiment, parameter)
    # Get target distributions
    dist_dict = get_distributions(experiment, interval, parameter)
    # Creating figure
    sns.set_style("whitegrid")
    sns.set_style("ticks", {"xtick.major.size": 5, "ytick.major.size": 5})

    fig = plt.subplot(rows, columns, i + 1)
    colors = []
    # Plot generated distributions
    for epoch_n in dist_dict.keys():
        angles = dist_dict[epoch_n]
        label = " ".join(epoch_n.split("_"))
        hist, edges = np.histogram(angles, bins=20)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        bin_widths = edges[1:] - edges[:-1]
        plt.bar(
            bin_centers,
            hist,
            width=bin_widths,
            linewidth=0.01,
            label=label,
            joinstyle="bevel",
            alpha=0.6,
        )
        plot = plt.plot(bin_centers, hist, linewidth=3, alpha=0.6)
        colors.append(plot[0].get_color())

    # Plot target distribution
    tgt_angles = tgt_dist_dict["tgt_dist"]
    hist, edges = np.histogram(tgt_angles, bins=20)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_widths = edges[1:] - edges[:-1]

    plt.bar(
        bin_centers,
        hist,
        width=bin_widths,
        linewidth=0.01,
        alpha=0.6,
        label="Target dist.",
        joinstyle="bevel",
    )
    plot = plt.plot(bin_centers, hist, linewidth=3, alpha=0.6)
    colors.append(plot[0].get_color())
    # Finalize plot
    plt.title(title, fontsize=16)
    plt.gcf().set_size_inches(size[0], size[1])
    plt.legend(loc="upper left", prop={"size": 15})
    plt.xlabel(f"{parameter}", fontsize=15)
    plt.xticks(fontsize=14)
    plt.ylabel("frequency", fontsize=15)
    plt.yticks(fontsize=14)

    return fig


def plot_each_distribution(experiment, interval, parameter="yaw"):
    # Plot joint plot and get colors
    colors = plot_distributions(experiment, interval, parameter)
    # Get target distribution
    tgt_dist_dict = get_tgt_distribution(experiment, parameter)
    # Get target distributions
    dist_dict = get_distributions(experiment, interval, parameter)
    dist_dict["Target dist"] = tgt_dist_dict["tgt_dist"]
    # Creating figure
    sns.set_style("whitegrid")
    sns.set_style("ticks", {"xtick.major.size": 5, "ytick.major.size": 5})

    dim = len(dist_dict)
    fig = plt.figure(figsize=(dim * 5, (dim * 5) / dim))
    plt.suptitle(
        f"Generated distributions for {parameter} parameter at each key epoch",
        fontweight="bold",
        y=1.05,
    )
    # Plot generated distributions
    for i, epoch_n in enumerate(dist_dict.keys()):
        ax = fig.add_subplot(1, dim, i + 1)
        angles = dist_dict[epoch_n]
        hist, edges = np.histogram(angles, bins=20)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        bin_widths = edges[1:] - edges[:-1]
        label = " ".join(epoch_n.split("_"))

        ax.bar(
            bin_centers,
            hist,
            width=bin_widths,
            alpha=0.7,
            label=label,
            color=colors[i],
            joinstyle="bevel",
            linewidth=0.01,
        )
        ax.plot(bin_centers, hist, linewidth=1, color=colors[i], alpha=0.7)

        # Finalize plot
        plt.title(f"{epoch_n}")
        plt.xlabel(f"{parameter}")

    plt.ylabel("frequency")
    plt.show()


def plot_each_image(experiment_n, interval, filename, title):
    # Get target distribution
    epoch_images = get_gen_images(experiment_n, interval)

    # Creating figure
    sns.set_style("whitegrid")
    sns.set_style("ticks", {"xtick.major.size": 5, "ytick.major.size": 5})

    dim = len(epoch_images)
    fig = plt.figure(figsize=(dim * 5, (dim * 5) / dim))
    plt.suptitle(title, fontweight="bold", fontsize=32, y=1.09)
    # Plot generated distributions
    for i, epoch_n in enumerate(epoch_images.keys()):
        ax = fig.add_subplot(1, dim, i + 1)
        plt.imshow(epoch_images[epoch_n], label=" ".join(epoch_n.split("_")))
        # Finalize plot
        # plt.title(' '.join(epoch_n.split('_')), fontsize=18)
        plt.text(
            0.01,
            0.93,
            " ".join(epoch_n.split("_")),
            fontsize=32,
            transform=ax.transAxes,
            backgroundcolor="1",
            alpha=1,
        )
        plt.tick_params(
            left=False, right=False, labelleft=False, labelbottom=False, bottom=False
        )

    plt.tight_layout()
    fig.savefig(
        os.path.join(RESULTS_PATH, f"{filename}.pdf"),
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.show()


def plot_initial_distributions(
    experiment_n=1, interval=4, parameter="yaw", title="No title", filename="noname.pdf"
):
    # Get target distribution
    tgt_dist_dict = get_tgt_distribution(experiment_n, parameter)
    # Get target distributions
    dist_dict = get_distributions(experiment_n, interval, parameter)
    dist_dict["Target dist."] = tgt_dist_dict["tgt_dist"]

    # Creating figure
    sns.set_style("whitegrid")
    sns.set_style("ticks", {"xtick.major.size": 5, "ytick.major.size": 5})

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(12, 2.5), gridspec_kw={"width_ratios": [7, 3, 3]}
    )
    fig.suptitle(title, fontweight="bold", fontsize=14, y=1.04)

    # ax1
    # Plot generated distributions
    for epoch_n in dist_dict.keys():
        angles = dist_dict[epoch_n]
        hist, edges = np.histogram(angles, bins=20)
        bin_centers = 0.5 * (edges[:-1] + edges[1:])
        bin_widths = edges[1:] - edges[:-1]
        if epoch_n == "Target dist.":
            ax1.bar(
                bin_centers,
                hist,
                width=bin_widths,
                label=" ".join(epoch_n.split("_")),
                color="tab:pink",
                joinstyle="bevel",
                alpha=0.7,
                linewidth=0.01,
            )
            ax1.plot(bin_centers, hist, color="tab:pink", linewidth=3, alpha=0.7)
        else:
            ax1.bar(
                bin_centers,
                hist,
                width=bin_widths,
                label=" ".join(epoch_n.split("_")),
                joinstyle="bevel",
                alpha=0.7,
                linewidth=0.01,
            )
            ax1.plot(bin_centers, hist, linewidth=3, alpha=0.7)

    # Finalize plot
    ax1.set_title("Genenerated distributions at each key epoch", fontsize=13)
    ax1.legend(
        loc="upper center",
        fontsize=11,
        ncol=1,
        bbox_to_anchor=(0.75, 1),
    )
    ax1.set_xlabel(f"{parameter}", fontsize=13)
    ax1.set_ylabel("frequency", fontsize=13)

    # ax1
    angles = dist_dict["epoch_0"]
    hist, edges = np.histogram(angles, bins=20)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_widths = edges[1:] - edges[:-1]
    label = " ".join(epoch_n.split("_"))

    ax2.bar(
        bin_centers,
        hist,
        width=bin_widths,
        alpha=0.7,
        color="tab:blue",
        joinstyle="bevel",
        linewidth=0.01,
    )
    ax2.plot(bin_centers, hist, linewidth=1, color="tab:blue", alpha=0.7)

    # Finalize plot
    ax2.set_title("Distribution at epoch 0", fontsize=13)
    ax2.set_xlabel("yaw", fontsize=13)

    # ax3
    angles = dist_dict["Target dist."]
    hist, edges = np.histogram(angles, bins=20)
    bin_centers = 0.5 * (edges[:-1] + edges[1:])
    bin_widths = edges[1:] - edges[:-1]

    ax3.bar(
        bin_centers,
        hist,
        width=bin_widths,
        alpha=0.7,
        color="tab:pink",
        joinstyle="bevel",
        linewidth=0.01,
    )
    ax3.plot(bin_centers, hist, linewidth=1, color="tab:pink", alpha=0.7)

    # Finalize plot
    ax3.set_title("Target distribution", fontsize=13)
    ax3.set_xlabel("yaw", fontsize=13)
    fig.tight_layout()
    fig.savefig(
        os.path.join(RESULTS_PATH, f"{filename}.pdf"),
        bbox_inches="tight",
        pad_inches=0.1,
    )


def plot_translation_distribution(experiment_n, interval):
    epochx_dictionnary = get_distributions(
        experiment_n=experiment_n, interv=interval, parameter="loc_x"
    )
    epochz_dictionnary = get_distributions(
        experiment_n=experiment_n, interv=interval, parameter="loc_z"
    )
    tgtx_dictionnary = get_tgt_distribution(
        experiment_n=experiment_n, parameter="loc_x"
    )
    tgtz_dictionnary = get_tgt_distribution(
        experiment_n=experiment_n, parameter="loc_z"
    )

    x_tgt = tgtx_dictionnary["tgt_dist"]
    z_tgt = tgtz_dictionnary["tgt_dist"]

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.subplots_adjust(left=0.1, right=0.875, top=0.9, bottom=0.125)

    for epoch in epochx_dictionnary.keys():
        x_gen = epochx_dictionnary[epoch]
        z_gen = epochz_dictionnary[epoch]
        ax.scatter(x_gen, z_gen, label=f"{epoch}", alpha=0.25)

    ax.scatter(x_tgt, z_tgt, alpha=0.5, label="Target")

    ax.set_xlabel("loc_x", rotation=0, fontsize=12)
    ax.invert_xaxis()
    # ax.xaxis.tick_top()
    ax.xaxis.set_label_position("bottom")
    ax.set_ylabel("loc_z", rotation=0, fontsize=12)
    ax.invert_yaxis()
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    plt.setp(ax.get_xticklabels(), rotation=0, va="top", ha="center")
    plt.setp(ax.get_yticklabels(), rotation=0, va="center", ha="left")

    ax.text(
        0.5,
        1.06,
        "Translation distribution",
        ha="center",
        va="center",
        transform=ax.transAxes,
        rotation=0,
        fontweight="bold",
        fontsize=14,
    )

    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def create_locations_df(experiment_n, interval):
    epochx_dictionnary = get_distributions(
        experiment_n=experiment_n, interv=interval, parameter="loc_x"
    )
    epochz_dictionnary = get_distributions(
        experiment_n=experiment_n, interv=interval, parameter="loc_z"
    )
    tgtx_dictionnary = get_tgt_distribution(
        experiment_n=experiment_n, parameter="loc_x"
    )
    tgtz_dictionnary = get_tgt_distribution(
        experiment_n=experiment_n, parameter="loc_z"
    )
    epochx_dictionnary["Target dist."] = tgtx_dictionnary["tgt_dist"]
    epochz_dictionnary["Target dist."] = tgtz_dictionnary["tgt_dist"]

    dict_df = {"loc x": [], "loc z": [], "Epoch": []}

    for epoch in epochx_dictionnary.keys():
        dict_df["loc x"] = dict_df["loc x"] + epochx_dictionnary[epoch]
        dict_df["loc z"] = dict_df["loc z"] + epochz_dictionnary[epoch]
        dict_df["Epoch"] = dict_df["Epoch"] + [" ".join(epoch.split("_"))] * len(
            epochx_dictionnary[epoch]
        )

    return pd.DataFrame.from_dict(dict_df)


def create_camrotations_df(experiment_n, interval):
    epochx_dictionnary = get_distributions(
        experiment_n=experiment_n, interv=interval, parameter="camera_pitch"
    )
    epochz_dictionnary = get_distributions(
        experiment_n=experiment_n, interv=interval, parameter="camera_roll"
    )
    tgtx_dictionnary = get_tgt_distribution(
        experiment_n=experiment_n, parameter="camera_pitch"
    )
    tgtz_dictionnary = get_tgt_distribution(
        experiment_n=experiment_n, parameter="camera_roll"
    )
    epochx_dictionnary["Target dist."] = tgtx_dictionnary["tgt_dist"]
    epochz_dictionnary["Target dist."] = tgtz_dictionnary["tgt_dist"]

    dict_df = {"camera pitch": [], "camera roll": [], "Epoch": []}

    for epoch in epochx_dictionnary.keys():
        dict_df["camera pitch"] = dict_df["camera pitch"] + epochx_dictionnary[epoch]
        dict_df["camera roll"] = dict_df["camera roll"] + epochz_dictionnary[epoch]
        dict_df["Epoch"] = dict_df["Epoch"] + [" ".join(epoch.split("_"))] * len(
            epochx_dictionnary[epoch]
        )

    return pd.DataFrame.from_dict(dict_df)


def plot_loc_dist_grid(experiment_n, height, title, filename, legend, ylabel):
    loc_df = create_locations_df(experiment_n=experiment_n, interval=2)
    jointgrid = sns.jointplot(
        data=loc_df,
        x="loc z",
        y="loc x",
        hue="Epoch",
        height=height,
        alpha=0.25,
        legend=legend,
    )
    jointgrid.fig.axes[0].invert_xaxis()
    jointgrid.fig.axes[0].invert_yaxis()
    if not ylabel:
        jointgrid.set_axis_labels("loc z", None, fontsize=13)
    else:
        jointgrid.set_axis_labels("loc z", "loc x", fontsize=13)
    jointgrid.fig.suptitle(title, fontsize=14, y=0.95)
    jointgrid.fig.tight_layout()
    jointgrid.fig.subplots_adjust(top=0.9, right=0.9)
    jointgrid.savefig(os.path.join(RESULTS_PATH, f"{filename}.pdf"))

    return jointgrid


def plot_camera_dist_grid(
    experiment_n, interval, height, filename, title, legend, ylabel
):
    loc_df = create_camrotations_df(experiment_n=experiment_n, interval=interval)
    jointgrid = sns.jointplot(
        data=loc_df,
        x="camera pitch",
        y="camera roll",
        height=height,
        hue="Epoch",
        alpha=0.25,
    )
    if not ylabel:
        jointgrid.set_axis_labels("camera pitch", None, fontsize=13)
    else:
        jointgrid.set_axis_labels("camera pitch", "camera roll", fontsize=13)
    jointgrid.fig.suptitle(title, fontsize=14, y=1)
    jointgrid.fig.tight_layout()
    jointgrid.fig.subplots_adjust(top=0.95, right=0.95)
    jointgrid.savefig(os.path.join(RESULTS_PATH, f"{filename}.pdf"))


def create_3D_rotation(exp_list, interval, title, sub_titles, filename):
    # Define basic information
    dim = len(exp_list)
    sns.set_style("whitegrid")
    sns.set_style("ticks", {"xtick.major.size": 2, "ytick.major.size": 2})
    fig = plt.figure(figsize=(dim * 4.2, dim * 4 / 3))
    plt.suptitle(title, fontweight="bold", fontsize=17, y=1.04)

    for i, experiment_n in enumerate(exp_list):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        # Get distributions
        dist_dict_yaw = get_distributions(
            experiment_n=experiment_n, interv=interval, parameter="yaw"
        )
        dist_dict_pitch = get_distributions(
            experiment_n=experiment_n, interv=interval, parameter="pitch"
        )
        dist_dict_roll = get_distributions(
            experiment_n=experiment_n, interv=interval, parameter="roll"
        )
        tgt_dict_yaw = get_tgt_distribution(experiment_n=experiment_n, parameter="yaw")
        tgt_dict_pitch = get_tgt_distribution(
            experiment_n=experiment_n, parameter="pitch"
        )
        tgt_dict_roll = get_tgt_distribution(
            experiment_n=experiment_n, parameter="roll"
        )
        # Get dataframe
        dist_dict_yaw["Target_dist."] = tgt_dict_yaw["tgt_dist"]
        dist_dict_pitch["Target_dist."] = tgt_dict_pitch["tgt_dist"]
        dist_dict_roll["Target_dist."] = tgt_dict_roll["tgt_dist"]

        for e, epoch in enumerate(dist_dict_yaw.keys()):
            # Get components
            pitch, yaw, roll = (
                dist_dict_pitch[epoch],
                dist_dict_yaw[epoch],
                dist_dict_roll[epoch],
            )
            # Creating plot
            ax.scatter(
                pitch,
                yaw,
                roll,
                label=" ".join(epoch.split("_")),
                alpha=0.5,
                s=75,
                linewidths=1.5,
                edgecolor="white",
            )

        plt.title(sub_titles[i], fontsize=16)
        ax.set_xlabel("pitch (x)", fontsize=13)
        ax.set_ylabel("yaw (y)", fontsize=13)
        ax.set_zlabel("roll (z)", fontsize=13)

        if i == 2:
            plt.legend(loc="center", fontsize=12, bbox_to_anchor=(1.3, 0.5))

    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.1)
    fig.savefig(
        os.path.join(RESULTS_PATH, f"{filename}.pdf"),
        bbox_inches="tight",
        pad_inches=0.2,
    )