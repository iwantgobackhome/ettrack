from pytracking.evaluation import Tracker, get_dataset, trackerlist


def et_tracker():
    trackers = trackerlist('et_tracker', 'et_tracker', range(1))

    # dataset = get_dataset('got10k_test')
    dataset = get_dataset('uav')
    return trackers, dataset

