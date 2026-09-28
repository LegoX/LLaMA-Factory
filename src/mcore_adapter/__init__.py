# Copyright 2025 the ROLL team and the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Vendored from https://github.com/alibaba/ROLL (mcore_adapter, commit 192b1a01ea61c113b2deb543f7b115783038dff8).
# Local changes are listed in examples/megatron/README_512k.md.

from .models import McaGPTModel, McaModelConfig
from .trainer import McaTrainer
from .training_args import Seq2SeqTrainingArguments, TrainingArguments


__all__ = ["McaModelConfig", "McaGPTModel", "TrainingArguments", "Seq2SeqTrainingArguments", "McaTrainer"]
