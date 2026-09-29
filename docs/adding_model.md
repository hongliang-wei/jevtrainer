# Adding a model

Most new checkpoints need nothing: set `model:` to the Hub id or a local path and `family: auto`.
`GenericFamily` infers

* the model class: `AutoModelForImageTextToText` if the config has a `vision_config`, else `AutoModelForCausalLM`;
* the backbone (`model.model`) and LM head (`get_output_embeddings()`);
* LoRA targets: every `nn.Linear` whose name contains `proj`, `fc`, `dense`, … outside the vision tower and LM head;
* the vision tower (modules named `visual`, `vision_tower`, `multi_modal_projector`, …), frozen by default;
* whether the model has linear-attention / recurrent layers.

Check what it infers without loading weights:

```bash
jt model inspect Qwen/Qwen3.5-0.8B
```

If something is wrong, add a family. Copy one of `jevtrainer/model/families/others.py` and override
only what differs:

```python
from jevtrainer.model.families.base import ModelFamily
from jevtrainer.registry import FAMILIES

@FAMILIES.register("newmodel")
class NewFamily(ModelFamily):
    name = "newmodel"
    model_types = ("newmodel", "newmodel_vl")          # config.model_type values

    def image_placeholder(self, processor):            # used by the pointer readout
        return "<image>"

    def forward_kwargs(self, model, batch):            # extra backbone kwargs (positions, pixels)
        return super().forward_kwargs(model, batch)
```

Qwen VL models are the example of a real override: because readouts pass `inputs_embeds`, the
family computes M-RoPE `position_ids` itself (`qwen.py`).

Encoders (ModernBERT, DeBERTa) use `encoder.py`: no LM head, bidirectional attention, so they work
with `slot` and `pointer` but not `marker`.

Test a new family with a random tiny config in `tests/test_families.py` (see `test_llama_generic_path`):
build each readout, run LoRA, check the logits shapes.
