

# GCN helper functions
def get_GCN_progress(experiment_n):
    progress_path = os.path.join(ROOT_PATH, f'experiment{experiment_n}', PROG_PATH)
    return import_csv(os.path.join(progress_path, 'gcn.csv'))

def plot_pretraining_progress(experiment_n, title, filename, size):
    progress_path = os.path.join(ROOT_PATH, f'experiment{experiment_n}', PROG_PATH)
    gcn_progres = import_csv(os.path.join(progress_path, 'gcn.csv'))
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=size)
    plt.plot(gcn_progres['rec_epoch'], gcn_progres['class_loss'], label='classification loss', linewidth=3, alpha=0.8)
    plt.plot(gcn_progres['rec_epoch'], gcn_progres['cont_loss'], label='content loss', linewidth=3, alpha=0.8)
    plt.plot(gcn_progres['rec_epoch'], gcn_progres['loss'], label='total loss', linewidth=3, alpha=0.8)
    plt.legend(fontsize=15)
    plt.title(title, fontweight='bold', fontsize=15)
    plt.xlabel("epoch", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.ylabel("loss", fontsize=15)
    sns.despine()
    fig.savefig(os.path.join(RESULTS_PATH, f'{filename}.pdf'), bbox_inches='tight')

def plot_pretraining_progress_grid(experiment_n, title, i):
    progress_path = os.path.join(ROOT_PATH, f'experiment{experiment_n}', PROG_PATH)
    gcn_progres = import_csv(os.path.join(progress_path, 'gcn.csv'))
    sns.set_style("white")
    fig = plt.subplot(1, 3, i+1)
    plt.plot(gcn_progres['rec_epoch'], gcn_progres['class_loss'], label='class loss')
    plt.plot(gcn_progres['rec_epoch'], gcn_progres['cont_loss'],  label='cont loss')
    plt.plot(gcn_progres['rec_epoch'], gcn_progres['loss'],       label='(total) loss')
    plt.legend(fontsize=12)
    plt.title(title, fontsize=15)
    plt.xlabel("epoch", fontsize=13)
    plt.ylabel("loss", fontsize=13)
    plt.gcf().set_size_inches(15, 3)

    return fig

def get_epoch_graphs(graph_path, epoch):
    graph_epoch_path = os.path.join(graph_path, f'epoch_{epoch}')
    return walk_npy_folder(graph_epoch_path)

def get_feature_graphs(experiment_n, dimension, interv):
    graph_path = os.path.join(ROOT_PATH, f'experiment{experiment_n}', GRAPH_PATH)
    hpams = read_hyperaparameters(experiment_n)
    total_epochs = hpams['max_epochs']
    freq = hpams['distprint_freq']
    interval = interv

    feature_dict = {}
    for e in range(0, total_epochs+1, freq*interval):
        graph_features = get_epoch_graphs(graph_path, e)
        if len(graph_features) > 1:
            graph_features = np.reshape(graph_features, \
                (graph_features.shape[0]*graph_features.shape[1], graph_features.shape[2]))
            feature_dict[f'epoch_{e}'] =  pca_array(graph_features, \
                                                    components=dimension).T
            
        else:
            print("Full results not ready yet")
            break

    return feature_dict

def plot_gfeature_representation(experiment, components, interval, filename):
    # Get all graph features in 3D representation
    graph_features = get_feature_graphs(experiment_n=experiment, dimension=components, interv=interval)

    # Creating figure
    sns.set_style('whitegrid')
    sns.set_style("ticks", {"xtick.major.size": 2, "ytick.major.size": 2})

    dim = len(graph_features)
    fig = plt.figure(figsize = (dim*3, (dim*3)/dim))
    suptitle = plt.suptitle("Graph batch node feature representation", fontweight="bold", fontsize=20, y=1.05)
    for i, epoch in enumerate(graph_features.keys()):
        # Get epoch's data
        node_features = graph_features[epoch]
        epoch = int(epoch.split('_')[-1])
        # Create figure
        # Creating plot
        if node_features.shape[0] == 3:
            ax = fig.add_subplot(1, dim, i+1, projection='3d')
            ax.scatter3D(node_features[0], node_features[1], node_features[2],\
                color = 'teal', s=40, edgecolors='white', linewidths=0.8, alpha=0.0025)
            plt.title(f"epoch {epoch}", fontsize=17)
            plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)

        else:
            ax = fig.add_subplot(1, dim, i+1)
            ax.scatter(node_features[0], node_features[1],\
                color = 'teal', s=40, edgecolors='white', linewidths=0.8, alpha=0.0025)
            plt.title(f"epoch {epoch}", fontsize=17)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_PATH, f'{filename}.png'), bbox_inches='tight', pad_inches=0.2)