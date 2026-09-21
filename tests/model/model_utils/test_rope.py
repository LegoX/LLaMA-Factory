# Copyright 2026 the LlamaFactory team.
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

from types import SimpleNamespace

from llamafactory.extras.constants import RopeScaling
from llamafactory.model.model_utils.rope import configure_rope


def test_qwen3_5_nested_yarn_rope_scaling():
    rope_parameters = {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "rope_type": "default",
        "rope_theta": 10_000_000,
        "partial_rotary_factor": 0.25,
    }
    text_config = SimpleNamespace(
        max_position_embeddings=262_144,
        rope_scaling=rope_parameters,
        rope_parameters=rope_parameters,
    )
    config = SimpleNamespace(model_type="qwen3_5_moe", text_config=text_config)
    model_args = SimpleNamespace(rope_scaling=RopeScaling.YARN, model_max_length=524_288)

    configure_rope(config, model_args)

    assert text_config.max_position_embeddings == 524_288
    assert text_config.rope_scaling == text_config.rope_parameters
    assert text_config.rope_scaling["rope_type"] == "yarn"
    assert text_config.rope_scaling["factor"] == 2.0
    assert text_config.rope_scaling["original_max_position_embeddings"] == 262_144
    assert text_config.rope_scaling["rope_theta"] == 10_000_000
    assert text_config.rope_scaling["partial_rotary_factor"] == 0.25
    assert text_config.rope_scaling["mrope_section"] == [11, 11, 10]
