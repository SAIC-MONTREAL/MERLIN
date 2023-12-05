from pathlib import Path
from collections import defaultdict
import shutil
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from tensorflow.core.util import event_pb2
from torch.utils.tensorboard import SummaryWriter

class SummaryReader:
    """
    Util to read tensorboard log files.

    For now, read everything at once, but in the future we may want to
    add an iterator version if memory becomes an issue.
    """
    @staticmethod
    def get_readers(directory, fields=None):
        """
        Get a list of SummaryReaders, one for each event file found (recursively) in
        directory.
        """
        return [SummaryReader(path, fields=fields) for path in directory.rglob('events.out*')]


    def __init__(self, path, fields=None):
        """
        Args:
            path (str): path to log file or directory with a single log file
                in it.
            fields (list of str): list of selected fields to read, if None read
                all.
        """
        if isinstance(path, str):
            path = Path(path)
        if 'events.out' in path.stem:
            self.path = path
        elif path.is_dir():
            # just find one
            path = next(path.rglob('events.out*'))
        else:
            raise ValueError('Invalid path')
        self.path = path
        self.fields = fields

    def _read_event(self, event):
        out = {'step': event.step}
        for value in event.summary.value:
            if self.fields is not None and value.tag not in self.fields:
                continue
            # This has been tested when writer is pytorch SummaryWriter
            # and field is scalar
            if hasattr(value, 'simple_value'):
                out[value.tag] = value.simple_value

            # This kind of stuff worked with tf2 based writer, and contains code
            # for images. I'm leaving it as a comment for reference, but we will need
            # to do some more work to make this work properly.
            #  t = tf.make_ndarray(value.tensor)
            #  if value.tag == 'best_img':
                #  t = tf.io.decode_png(t[2])
                #  best_img = t
                #  continue
            #  t = t[()]
            #  if value.tag == 'alpha_0.0_total_evals':
                #  timeseries['total_evals'].append(t)
        return out

    def read(self):
        dataset = tf.data.TFRecordDataset(str(self.path))
        events = defaultdict(lambda: {})
        for event_ser in dataset:
            event = event_pb2.Event.FromString(event_ser.numpy())
            events[event.step].update(self._read_event(event))
        return pd.DataFrame([v for k, v in sorted(events.items())])


@pytest.yield_fixture(scope='function')
def temp_dir():
    td = mkdtemp()
    yield td
    shutil.rmtree(td)


def test_summary_reader(temp_dir):
    writer = SummaryWriter(log_dir=temp_dir)
    writer.add_scalar('scalar', 42, 0)
    writer.add_scalar('scalar2', 1, 0)
    writer.add_scalar('scalar', 43, 1)
    writer.flush()
    reader = SummaryReader(temp_dir)
    data = reader.read()
