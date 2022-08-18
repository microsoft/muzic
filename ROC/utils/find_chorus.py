import numpy as np
import scipy
import scipy.signal
LINE_THRESHOLD = 0.12
NUM_ITERATIONS = 8
MIN_LINES = 8
OVERLAP_PERCENT_MARGIN = 0.2
class Line(object):
    def __init__(self, start, end, lag):
        self.start = start
        self.end = end
        self.lag = lag

    def __repr__(self):
        return "Line ({} {} {})".format(self.start, self.end, self.lag)

def detect_lines_helper(denoised_time_lag, rows, threshold,
                        min_length_samples):
    """Detect lines where at least min_length_samples are above threshold"""
    num_samples = denoised_time_lag.shape[0]
    line_segments = []
    cur_segment_start = None
    for row in rows:
        if row < min_length_samples:
            continue
        for col in range(row, num_samples):
            if denoised_time_lag[row, col] > threshold:
                if cur_segment_start is None:
                    cur_segment_start = col
            else:
                if cur_segment_start is not None and (col - cur_segment_start) > min_length_samples:
                    line_segments.append(Line(cur_segment_start, col, row))
                cur_segment_start = None
    return line_segments


def detect_lines(denoised_time_lag, rows, min_length_samples):
    """Detect lines in the time lag matrix. Reduce the threshold until we find enough lines"""

    cur_threshold = LINE_THRESHOLD
    for _ in range(NUM_ITERATIONS):
        line_segments = detect_lines_helper(denoised_time_lag, rows,
                                            cur_threshold, min_length_samples)
        if len(line_segments) >= MIN_LINES:
            return line_segments
        cur_threshold *= 0.95

    return line_segments

def local_maxima_rows(denoised_time_lag):
    """Find rows whose normalized sum is a local maxima"""
    row_sums = np.sum(denoised_time_lag, axis=1)
    divisor = np.arange(row_sums.shape[0], 0, -1)
    normalized_rows = row_sums / divisor
    local_minima_rows = scipy.signal.argrelextrema(normalized_rows, np.greater)
    return local_minima_rows[0]

def denoise(time_time_matrix, smoothing_size):
    """
    Emphasize horizontal lines by suppressing vertical and diagonal lines. We look at 6
    moving averages (left, right, up, down, upper diagonal, lower diagonal). For lines, the
    left or right average should be much greater than the other ones.

    Args:
        time_time_matrix: n x n numpy array to quickly compute diagonal averages
        smoothing_size: smoothing size in samples (usually 1-2 sec is good)
    """
    n = time_time_matrix.shape[0]

    # Get the horizontal strength at every sample
    horizontal_smoothing_window = np.ones(
        (1, smoothing_size)) / smoothing_size
    horizontal_moving_average = scipy.signal.convolve2d(
        time_time_matrix, horizontal_smoothing_window, mode="full")
    left_average = horizontal_moving_average[:, 0:n]
    right_average = horizontal_moving_average[:, smoothing_size - 1:]
    max_horizontal_average = np.maximum(left_average, right_average)

    # Get the vertical strength at every sample
    vertical_smoothing_window = np.ones((smoothing_size,
                                         1)) / smoothing_size
    vertical_moving_average = scipy.signal.convolve2d(
        time_time_matrix, vertical_smoothing_window, mode="full")
    down_average = vertical_moving_average[0:n, :]
    up_average = vertical_moving_average[smoothing_size - 1:, :]

    # Get the diagonal strength of every sample from the time_time_matrix.
    # The key insight is that diagonal averages in the time lag matrix are horizontal
    # lines in the time time matrix
    diagonal_moving_average = scipy.signal.convolve2d(
        time_time_matrix, horizontal_smoothing_window, mode="full")
    ur_average = np.zeros((n, n))
    ll_average = np.zeros((n, n))
    for x in range(n):
        for y in range(x):
            ll_average[y, x] = diagonal_moving_average[x - y, x]
            ur_average[y, x] = diagonal_moving_average[x - y,
                                                       x + smoothing_size - 1]

    non_horizontal_max = np.maximum.reduce([down_average, up_average, ll_average, ur_average])
    non_horizontal_min = np.minimum.reduce([up_average, down_average, ll_average, ur_average])

    # If the horizontal score is stronger than the vertical score, it is considered part of a line
    # and we only subtract the minimum average. Otherwise subtract the maximum average
    suppression = (max_horizontal_average > non_horizontal_max) * non_horizontal_min + (
        max_horizontal_average <= non_horizontal_max) * non_horizontal_max

    # Filter it horizontally to remove any holes, and ignore values less than 0
    denoised_matrix = scipy.ndimage.filters.gaussian_filter1d(
        np.triu(time_time_matrix - suppression), smoothing_size, axis=1)
    denoised_matrix = np.maximum(denoised_matrix, 0)
    #denoised_matrix[0:5, :] = 0

    return denoised_matrix


def count_overlapping_lines(lines, margin, min_length_samples):
    """Look at all pairs of lines and see which ones overlap vertically and diagonally"""
    line_scores = {}
    for line in lines:
        line_scores[line] = 0

    # Iterate over all pairs of lines
    for line_1 in lines:
        for line_2 in lines:
            # If line_2 completely covers line_1 (with some margin), line_1 gets a point
            lines_overlap_vertically = (
                line_2.start < (line_1.start + margin)) and (
                    line_2.end > (line_1.end - margin)) and (
                        abs(line_2.lag - line_1.lag) > min_length_samples)

            lines_overlap_diagonally = (
                (line_2.start - line_2.lag) < (line_1.start - line_1.lag + margin)) and (
                    (line_2.end - line_2.lag) > (line_1.end - line_1.lag - margin)) and (
                        abs(line_2.lag - line_1.lag) > min_length_samples)

            if lines_overlap_vertically or lines_overlap_diagonally:
                line_scores[line_1] += 1

    return line_scores

def best_segment(line_scores):
    """Return the best line, sorted first by chorus matches, then by duration"""
    lines_to_sort = []
    for line in line_scores:
        lines_to_sort.append((line, line_scores[line], line.end - line.start, line.start))

    lines_to_sort.sort(key=lambda x: (x[2], x[1]), reverse=True)
    # best_tuple = lines_to_sort[0]
    lines = lines_to_sort
    return lines[:10]


def Find_Chorus(SSM, length, clip_length_samples = 25):
    SSM = denoise(SSM, 4)
    candidate_rows = local_maxima_rows(SSM)
    for _ in range(NUM_ITERATIONS):
        lines = detect_lines(SSM, candidate_rows, clip_length_samples)
        if len(lines) == 0:
            clip_length_samples = int(clip_length_samples * 0.8)
        else:
            break
    line_scores = count_overlapping_lines(
        lines, OVERLAP_PERCENT_MARGIN * clip_length_samples,
        clip_length_samples)
    best_chorus = best_segment(line_scores)
    best_chorus.sort(key=lambda x: x[3])
    idx = 0
    while True:
        if idx + 1 >= len(best_chorus):
            break
        if best_chorus[idx][0].end - best_chorus[idx][0].start < length/4 and best_chorus[idx][0].end + clip_length_samples > best_chorus[idx + 1][0].start:
            best_chorus[idx][0].end = best_chorus[idx + 1][0].end
            del best_chorus[idx + 1]
            idx -= 1
        idx += 1

    while len(best_chorus) > 1:
        if best_chorus[0][0].start < 1/5 * length:
            del best_chorus[0]
        else:
            break
    if len(best_chorus) > 0:
        return best_chorus[0][0]
    else:
        return None