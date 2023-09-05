# Gradients and MMD helper functions
def plot_MMD_loss(experiment_n):
    progress_path = os.path.join(ROOT_PATH, f"experiment{experiment_n}", PROG_PATH)
    gcn_progres = import_csv(os.path.join(progress_path, "mmd.csv"))
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(7, 5))
    plt.plot(
        gcn_progres["epoch"], gcn_progres["mmd_loss"], color="teal", label="MMD loss"
    )
    plt.legend(fontsize=12)
    plt.title(f"MMD loss after training", fontweight="bold", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.show()


def plot_MMD_loss_joint(experiment_n, label, limit=None):
    progress_path = os.path.join(ROOT_PATH, f"experiment{experiment_n}", PROG_PATH)
    gcn_progres = import_csv(os.path.join(progress_path, "mmd.csv"))
    plt.plot(
        gcn_progres["epoch"][:limit],
        gcn_progres["mmd_loss"][:limit],
        label=label,
        alpha=0.8,
        linewidth=3,
    )
    plt.legend(fontsize=15, loc="lower center")


def plot_gradients(experiment_n, interval):
    gradient_path = os.path.join(ROOT_PATH, f"experiment{experiment_n}", GRAD_PATH)
    sns.set_style("white")
    sns.despine(left=True, bottom=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    for epoch in range(0, 11, interval):
        epoch_grad_path = os.path.join(gradient_path, f"grads_e_{epoch}.csv")
        epoch_grad_df = pd.read_csv(epoch_grad_path)
        # layer_names = ['_'.join(layer_name.split('.')[:]) for layer_name in epoch_grad_df['layer']]
        layer_names = [
            "_".join(layer_name.split(".")[3:]) + f"_{i}"
            for i, layer_name in enumerate(epoch_grad_df["layer"])
        ]
        batches = list(epoch_grad_df.columns[1:])
        # change slice to 23 when not wanting to print encoder gradients
        slice_ = 23
        plt.plot(
            layer_names[slice_:],
            epoch_grad_df[slice_:].sum(axis=1),
            color="teal",
            alpha=(1 - (0.1 * epoch)),
            label=f"epoch {epoch}",
        )
        plt.xticks(rotation=45, fontsize=8)
    sns.despine(left=True, bottom=True)
    plt.title("GCN decoder gradient flow", fontweight="bold", fotnsize=15)
    plt.legend()
    plt.show()


def plot_gradients_grid(
    experiment_n,
    rows,
    columns,
    size,
    interval,
    limit,
    title,
    alpha_mult,
    legend_none,
    i,
):
    gradient_path = os.path.join(ROOT_PATH, f"experiment{experiment_n}", GRAD_PATH)
    sns.set_style("white")
    fig = plt.subplot(rows, columns, i + 1)
    for epoch in range(0, limit, interval):
        epoch_grad_path = os.path.join(gradient_path, f"grads_e_{epoch}.csv")
        epoch_grad_df = pd.read_csv(epoch_grad_path)
        # layer_names = ['_'.join(layer_name.split('.')[:]) for layer_name in epoch_grad_df['layer']]
        layer_names = [
            "_".join(layer_name.split(".")[4:]) + f"_{i}"
            for i, layer_name in enumerate(epoch_grad_df["layer"])
        ]
        batches = list(epoch_grad_df.columns[1:])
        # change slice to 23 when not wanting to print encoder gradients
        slice_ = 23
        plt.plot(
            layer_names[slice_:],
            epoch_grad_df[slice_:].sum(axis=1),
            color="teal",
            alpha=(1 - (alpha_mult) * epoch),
            label=f"epoch {epoch}",
            linewidth=2.5,
        )
        plt.xticks(rotation=90, fontsize=15)
    sns.despine(left=True, bottom=True)
    plt.gcf().set_size_inches(size[0], size[1])
    plt.title(title, fontsize=17)
    if not legend_none:
        plt.legend(loc="upper left", fontsize=15, ncol=2)
    return fig