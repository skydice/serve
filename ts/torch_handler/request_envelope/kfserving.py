"""
The KFServing Envelope is used to handle the KFServing
Input Request inside Torchserve.
"""
import json
import logging
from itertools import chain

from .base import BaseEnvelope

logger = logging.getLogger(__name__)


class KFservingEnvelope(BaseEnvelope):
    """
    This function is used to handle the input request specified in KFServing
    format and converts it into a Torchserve readable format.

    Args:
        data - List of Input Request in KFServing Format
    Returns:
        [list]: Returns the list of the Input Request in Torchserve Format
    """
    _lengths = []

    def parse_input(self, data):
        lengths, batch = self._batch_from_kf(data)
        self._lengths = lengths
        return batch

    def format_output(self, data):
        """
        Returns the prediction response and captum explanation response of the input request.

        Args:
            outputs (List): The outputs arguments is in the form of a list of dictionaries.

        Returns:
            (list): The response is returned as a list of predictions and explanations
        """
        return self._batch_to_data(data, self._lengths)

    def _batch_from_kf(self, data_rows):
        mini_batches = [self._from_kf(data_row) for data_row in data_rows]
        lengths = [len(mini_batch) for mini_batch in mini_batches]
        full_batch = list(chain.from_iterable(mini_batches))
        return lengths, full_batch

    def _from_kf(self, data):
        rows = (data.get('data') or data.get('body') or data)['instances']
        for row_i, row in enumerate(rows):
            if list(row.keys()) == ['data']:
                if isinstance(row['data'], (bytes, bytearray)):
                    rows[row_i] = json.loads(row['data'].decode())
        return rows

    def _batch_to_kf(self, batch, lengths):
        outputs = []
        cursor = 0
        for length in lengths:
            cursor_end = cursor + length

            mini_batch = batch[cursor:cursor_end]
            outputs.append(self._to_kf(mini_batch))

            cursor = cursor_end
        return outputs

    def _to_kf(self, output):
        if not self._is_explain():
            out_dict = {
                'predictions': output
            }
        else:
            out_dict = {
                'explanations': output
            }
        return out_dict

    def _is_explain(self):
        if self.context and self.context.get_request_header(0, "explain"):
            if self.context.get_request_header(0, "explain") == "True":
                return True

        return False
