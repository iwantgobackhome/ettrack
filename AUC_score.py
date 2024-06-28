from pytracking.analysis.plot_results import *
from pytracking.evaluation import Tracker, get_dataset, trackerlist

trackers = trackerlist('et_tracker', 'et_tracker', range(1))

dataset = get_dataset('otb')
eval_data = check_and_load_precomputed_results(trackers, dataset, 'bike1')
tracker_names = eval_data['trackers']
valid_sequence = torch.tensor(eval_data['valid_sequence'], dtype=torch.bool)
ave_success_rate_plot_overlap = torch.tensor(eval_data['ave_success_rate_plot_overlap'])

# Index out valid sequences
auc_curve, auc = get_auc_curve(ave_success_rate_plot_overlap, valid_sequence)
print(auc)