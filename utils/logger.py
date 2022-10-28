# Code referenced from
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/04-utils/tensorboard/logger.py
import os
from io import BytesIO

import numpy as np
import scipy.misc
from PIL import Image

import tensorflow as tf


class Logger(object):

    def __init__(self, log_dir, name=None):
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, name))

    def scalar(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def image(self, tag, images, step):
        """Log a list of images."""
        # np.savez("../../images.npz", images)
        with self.writer.as_default():
            img_summaries = []
            for i, img in enumerate(images):
                # Write the image to a string
                # s = BytesIO()
                # Image.fromarray(img).save(s, format="png")
                # scipy.misc.toimage(img).save(s, format="png")
                if len(img.shape) == 3:
                    img_summaries.append(np.reshape(img, (img.shape[1], img.shape[2], -1)))
                else:
                    img_summaries.append(np.reshape(img, (img.shape[0], img.shape[1], -1)))
                # Create an Image object
                
                # img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                #                         height=img.shape[0],
                #                         width=img.shape[1])
                # Create a Summary value
                # img_summaries.append(tf.Summary.Value(
                #     tag='%s/%d' % (tag, i), image=img_sum))

            # Create and write Summary
            # summary = tf.Summary(value=img_summaries)
            tf.summary.image(tag, np.array(img_summaries), step)
            self.writer.flush()

    def histogram(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        with self.writer.as_default():
            counts, bin_edges = np.histogram(values, bins=bins)

            # Fill the fields of the histogram proto
            hist = tf.HistogramProto()
            hist.min = float(np.min(values))
            hist.max = float(np.max(values))
            hist.num = int(np.prod(values.shape))
            hist.sum = float(np.sum(values))
            hist.sum_squares = float(np.sum(values**2))

            # Drop the start of the first bin
            bin_edges = bin_edges[1:]

            # Add bin edges and counts
            for edge in bin_edges:
                hist.bucket_limit.append(edge)
            for c in counts:
                hist.bucket.append(c)

            # Create and write Summary
            tf.summary.histogram(tag, hist, step)
            # summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
            # self.writer.add_summary(summary, step)
            self.writer.flush()
