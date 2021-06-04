import os
from argparse import ArgumentParser

from src.utils.WindowEventsParser import WindowEventsParser
from datetime import datetime, timedelta


if __name__ == "__main__":
    arg_parser = ArgumentParser(description='.')
    arg_parser.add_argument('--dataset_path', type=str, required=True)
    arg_parser.add_argument('--slice_len', type=int, required=False, default=7)
    arg_parser.add_argument('--slice_offset', type=int, required=False, default=0)
    arg = arg_parser.parse_args()
    
    dataset_path = arg.dataset_path
    slice_len = arg.slice_len
    slice_offset = arg.slice_offset
    
    # data parser
    parser = WindowEventsParser()
    parser.read_data_from_file(dataset_path)
    all_events = parser.events

    start_date = datetime.strptime(all_events[0].date, "%Y-%m-%d")
    end_date = datetime.strptime(all_events[-1].date, "%Y-%m-%d")

    num_days = (end_date - start_date).days
    mid_date = start_date + timedelta(days=num_days // 2)

    period_start = mid_date + timedelta(days=-slice_len - slice_offset)
    period_end = mid_date + timedelta(days=slice_len - slice_offset)

    period_events = [ev for ev in all_events if
                     datetime.strptime(ev.date, '%Y-%m-%d') >= period_start and
                     datetime.strptime(ev.date, '%Y-%m-%d') <= period_end
                     ]

    source_file_name = os.path.splitext(os.path.basename(dataset_path))[0]
    out_file_name = source_file_name + "_sliced" + ".txt"
    out_file_path = "src" + os.path.sep + "data" + os.path.sep + out_file_name
    with open(out_file_path, "w+") as fp:
        for ev in period_events:
            event_line = " ".join([ev.date, ev.time, ev.sensor.location, ev.sensor.name, ev.sensor.state,
                                   ev.label])
            fp.write(event_line)
            fp.write("\n")
