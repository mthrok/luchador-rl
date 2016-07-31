from __future__ import absolute_import

import logging

import tensorflow as tf

_LG = logging.getLogger(__name__)


class _SummaryConfig(object):
    """Create placeholder and summary operations for the given tensors"""
    def __init__(self, summary_type, names):
        summary_funcs = {
            'scalar': tf.scalar_summary,
            'image': tf.image_summary,
            'audio': tf.audio_summary,
            'histogram': tf.histogram_summary,
        }
        func = summary_funcs[summary_type]

        self.ops, self.pfs = [], []
        for name in names:
            pf = tf.placeholder('float32')
            op = func(name, pf)
            self.ops.append(op)
            self.pfs.append(pf)


class SummaryWriter(object):
    def __init__(self, output_dir, graph=None):
        self.output_dir = output_dir

        self.summary_ops = {}
        self.placeholders = []
        self.writer = tf.train.SummaryWriter(self.output_dir)

    def init(self, **kwargs):
        pass

    def add_graph(self, graph=None, global_step=None):
        if graph:
            self.writer.add_graph(graph, global_step=global_step)

    def register(self, key, summary_type, names):
        self.summary_ops[key] = _SummaryConfig(summary_type, names)

    def summarize(self, key, global_step, values):
        cfg = self.summary_ops[key]
        feed_dict = {pf: value for pf, value in zip(cfg.pfs, values)}

        with tf.device('/cpu:0'):
            with tf.Session() as session:
                summaries = session.run(cfg.ops, feed_dict=feed_dict)
                for summary in summaries:
                    self.writer.add_summary(summary, global_step)
                self.writer.flush()