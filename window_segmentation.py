import pandas as pd

#NOTE: Overlap time < window_time
def get_segmented_list(eda, duration, window_time = 60, overlap_time = 30):
    size = int(eda.size/(duration/window_time))
    noverlap = int(eda.size/(duration/overlap_time))
    step = size - noverlap
    last_start = int(eda.size - size + 1)
    period_starts = range(0, int(last_start), int(step))
    reshaped_data = [eda.iloc[k:k + int(size)].reset_index(drop=True) for k in period_starts]
    return reshaped_data
