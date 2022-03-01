from labplatform.config import get_config

import tables as tl
import numpy as np
import os
from traits.api import HasTraits, Dict, List


class DataExplorer(HasTraits):

    subject_groups = Dict
    _all_subject_files = List

    def _get__all_subject_files(self):
        files = os.listdir(get_config('SUBJECT_ROOT'))
        return [sfs for sfs in files if '.h5' in sfs]

    def search(self, **kwargs):
        """
        search the database for certain dataset, up to experiments

        Args:
            **kwargs:

        Returns:

        """
        pass

    def _read_subject_files(self):
        pass
