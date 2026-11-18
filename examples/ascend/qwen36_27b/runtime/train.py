"""Snapshot entry for grouped-checkpoint continuation; baseline code is copied alongside."""

import os

import base_train as core
from grouped_checkpoint_candidate import install_grouped_checkpoint


_original_init = core.RealTrainer.__init__


def grouped_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    install_grouped_checkpoint(self.model, int(os.environ["PROBE_CHECKPOINT_GROUP"]))
    if self.args.process_index == 0:
        print("GROUPED_CHECKPOINT_ENABLED", os.environ["PROBE_CHECKPOINT_GROUP"], flush=True)


core.RealTrainer.__init__ = grouped_init
if __name__ == "__main__":
    core.workflow.CustomSeq2SeqTrainer = core.RealTrainer
    core.run_exp()
