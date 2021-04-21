from SEPGenerator import *
from CPDStatisticsGenerator import *

if __name__ == "__main__":
    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--file', type=str, required=True)
    arg_parser.add_argument('--window_length', type=int, required=True)
    arg_parser.add_argument('--N', type=int, required=True)

    # arguments matcher
    arg = arg_parser.parse_args()
    DATA_SET = arg.file
    WINDOW_LENGTH = arg.window_length
    N = arg.N

    # data parser
    parser = WindowEventsParser()
    parser.read_data_from_file(DATA_SET)
    all_events = parser.events

    # features
    # defines the list of features that will be extracted from each window
    features = build_features()
    feature_extractor = FeatureExtractor(features)

    feature_windows = []
    oneHotEncoder = Encoder()

    source_file_name = os.path.splitext(os.path.basename(DATA_SET))[0]
    dest_folder = "src" + os.path.sep + "results" + os.path.sep
    dest_file = dest_folder + source_file_name + ".pkl"

    if os.path.exists(dest_file):
        feature_windows = pickle.load(open(dest_file, "rb"))
    else:
        for i in range(0, len(all_events) - WINDOW_LENGTH + 1):
            print(i)
            # get current 30 events window
            window = Window(all_events[i:WINDOW_LENGTH + i])
            # get array of features from window
            feature_window = feature_extractor.extract_features_from_window(window)
            feature_windows.append(feature_window)

        pickle.dump(feature_windows, open(dest_file, "wb"))
    
    grid_search_folder = dest_folder + "grid_search" + os.path.sep
    kernel_param_grid = [0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 7.5, 10, 15, 20]
    regularization_param = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    
    for sigma in kernel_param_grid:
        for lamda in regularization_param:

            res_file_name = grid_search_folder + source_file_name + "_res_%3.2f_%3.2f" % (sigma, lamda)

            SEP = []

            for index in range(WINDOW_LENGTH, len(feature_windows) + 1 - N):
                # print('Index SEP: ' + str(index) + '/' + str(len(feature_windows) + 1 - N))
                previous_x = feature_windows[index - WINDOW_LENGTH : index - WINDOW_LENGTH + N]
                assert len(previous_x) == N

                current_x = feature_windows[index: N + index]
                assert len(current_x) == N

                # use previous_x as the Y samples for distribution of f_(t-1)(x) and
                # use current_x as the X samples for distribution f_t(x) in a call to densratio RuLSIF -
                densratio_res = densratio(x=np.array(current_x), y=np.array(previous_x), kernel_num=len(previous_x),
                                          sigma_range=[sigma], lambda_range=[lamda],
                                          verbose=False)

                g_sum = np.sum(densratio_res.compute_density_ratio(np.array(current_x))) / len(current_x)
                sep = max(0, 0.5 - g_sum)

                sensor_index = feature_windows.index(previous_x[N - 1]) + WINDOW_LENGTH

                add_statistics_to_dataset(sensor_index)

                SEP.append((round(sep, 4), sensor_index))

            save_sep_data(res_file_name, SEP)