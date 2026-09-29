# Adding a readout

A readout decides how a record is laid out as token rows and where option logits come from.
Subclass `jevtrainer.readouts.Readout` and register it:

```python
from jevtrainer.readouts.base import Read, Readout, option_text
from jevtrainer.registry import READOUTS

@READOUTS.register("mine")
class MyReadout(Readout):
    special_tokens = ["<ask>"]        # added to the tokenizer only; embeddings live in self.special
    max_options = 255

    def build(self, family, model):   # create parameters that need the hidden size
        self.head = torch.nn.Linear(family.hidden_size(model), 1)

    def encode(self, record, rng=None):          # -> list[Row]
        rows = []
        for name, q in record.questions.items():
            text = f"{self.truncate_state(record)}<ask>{self.clean(q.instructions)}"
            row = self.tokenize(text, record_images(record) if record.images else None)
            row.reads.append(Read(0, name, {"pos": self.positions(row.input_ids, "<ask>")[0]}))
            rows.append(row)
        return rows

    def score(self, model, hidden, batch):       # -> one 1-D tensor per read, in label order
        return [...]
```

What the base class does for you:

* `setup()` adds `special_tokens` to the tokenizer and creates `self.special`, a trainable embedding
  table; `embed()` swaps it in at those positions, so the model's vocabulary is never resized and the
  tokens train under LoRA too.
* `tokenize(text, images)` uses the processor when there are images (Qwen-VL M-RoPE positions are
  handled by the family) and the tokenizer otherwise.
* `collate()` pads rows and concatenates pixel values; `hidden()` runs the backbone with
  `inputs_embeds` and returns the last hidden states. Override `forward()` only if you need the
  full LM output.
* `clean(text)` stops user text from forging your special tokens; `truncate_state()` caps the state.

Return logits in the question's `labels()` order. The marker readout may append one extra entry
(the log-mass of the rest of the vocabulary) during training; losses handle a trailing extra entry,
and everything else slices `[:k]`.

Test it by adding the name to `READOUTS` in `tests/test_readouts.py`.
