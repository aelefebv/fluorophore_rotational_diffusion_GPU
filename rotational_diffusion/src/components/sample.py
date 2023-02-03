from rotational_diffusion.src import utils, components, np

class Sample:
    def __init__(self, fluorophore):
        self.fluorophore = fluorophore
        self.t_x = np.array([])
        self.t_y = np.array([])
        self.ratio_mean = []
        self.experiment = None

    def run_experiment(self, num_molecules, pulse_scheme, use_triplets=True):
        self.experiment = components.experiment.Experiment(self.fluorophore, num_molecules, triplet=use_triplets)
        pulse_scheme(self.experiment.fluorophores)

    def get_detector_counts(self, from_state, to_state, collection_times):
        x, y, _, t, = self.experiment.fluorophores.get_xyzt_at_transitions(from_state, to_state)
        t_collection = t[(t >= collection_times[0]) & (t <= collection_times[1])]
        x_collection = x[(t >= collection_times[0]) & (t <= collection_times[1])]
        y_collection = y[(t >= collection_times[0]) & (t <= collection_times[1])]
        t_x_temp, t_y_temp = utils.general.split_counts_xy(x_collection, y_collection, t_collection)
        self.t_x = np.concatenate([self.t_x, t_x_temp])
        self.t_y = np.concatenate([self.t_y, t_y_temp])
        if (len(t_x_temp) > 0) and (len(t_y_temp) > 0):
            self.ratio_mean.append(len(t_x_temp)/len(t_y_temp))
        elif len(t_x_temp) > 0:
            self.ratio_mean.append(0)
        else:
            self.ratio_mean.append(np.nan)
    # if len(mean_t_y) > 0:
        #     self.ratio_mean.append(len(mean_t_x)/len(mean_t_y))
        # else:
        #     self.ratio_mean.append(np.nan)
