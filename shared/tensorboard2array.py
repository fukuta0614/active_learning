from tensorflow.python.summary.event_accumulator import EventAccumulator
import numpy as np


def scalar2arrays(scalarEvents):
    """
    converts scalarEvent to set of numpy.array
    """
    wall_times = []
    steps = []
    values = []

    for event in scalarEvents:
        wall_times.append(event.wall_time)
        steps.append(event.step)
        values.append(event.value)

    return np.array(wall_times), np.array(steps), np.array(values)


accumulator = EventAccumulator('filename-of-event-file')
accumulator.Reload()  # load event files

wall_times, steps, values = scalar2arrays(accumulator.Scalars('name-of-summary'))
