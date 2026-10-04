---
license: apache-2.0
base_model: Qwen/Qwen3.5-0.8B
tags:
- jev
- marker
- qwen3.5
---

# jevtrainer-qwen35-0.8b-2026-10-02

A Jev-style decision model. One forward pass returns a probability for every option. It does not generate text. The readout is `marker`. The base `Qwen/Qwen3.5-0.8B` is fully fine-tuned, with the vision tower frozen. Training finished on **2026-10-02**.

The training config is [`configs/repro/intern_0.8b_v4.yaml`](https://github.com/hongliang-wei/jevtrainer/blob/main/configs/repro/intern_0.8b_v4.yaml) in [jevtrainer](https://github.com/hongliang-wei/jevtrainer): public text, Jev-format data, and images, about 540k training records, one epoch.

Weights are in `model/`. The same directory has `readout.safetensors`, `readout.json`, and `calibration.json`. `load()` reads those together. Pass a local directory; a Hub repo id is not accepted directly.

## Evaluation

Evaluation of `weihongliang/jevtrainer-qwen35-0.8b-2026-10-02`. Accuracy in percent.

Seven suites: Average is the unweighted mean of Easy, Original, Hard, Typed Decision, ToolACE, AG News, and WildJailBreak. longdoc-dev and gui-v1 are separate. Empty cells were not part of that comparison.

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="8">Seven suites. Average is the unweighted mean of these seven.</th>
<th rowspan="2">longdoc-dev</th>
<th rowspan="2">gui-v1</th>
</tr>
<tr>
<th>Easy</th>
<th>Original</th>
<th>Hard</th>
<th>Typed Decision</th>
<th>ToolACE</th>
<th>AG News</th>
<th>WildJailBreak</th>
<th>Average</th>
</tr>
</thead>
<tbody>
<tr>
<td><a href="https://docs.typesafe.ai/api">Jev</a></td>
<td align="right">100.00</td>
<td align="right">98.61</td>
<td align="right">72.07</td>
<td align="right">73.35</td>
<td align="right">91.29</td>
<td align="right">89.57</td>
<td align="right">96.29</td>
<td align="right">88.74</td>
<td></td>
<td></td>
</tr>
<tr>
<td><a href="https://github.com/NandhaKishorM/laya">Laya</a></td>
<td align="right">95.83</td>
<td align="right">72.22</td>
<td align="right">28.83</td>
<td align="right">35.95</td>
<td align="right">63.87</td>
<td align="right">92.84</td>
<td align="right">14.84</td>
<td align="right">57.77</td>
<td></td>
<td></td>
</tr>
<tr>
<td><a href="https://github.com/TheoLeeCJ/SemIf-OpenJev">SemIf</a></td>
<td align="right">100.00</td>
<td align="right">98.61</td>
<td align="right">61.26</td>
<td align="right">62.80</td>
<td align="right">85.16</td>
<td align="right">89.22</td>
<td align="right">92.53</td>
<td align="right">84.23</td>
<td></td>
<td></td>
</tr>
<tr>
<td><a href="https://github.com/jaredpalmer/kev">Kev</a></td>
<td align="right">100.00</td>
<td align="right">93.06</td>
<td align="right">45.05</td>
<td align="right">65.60</td>
<td align="right">87.42</td>
<td align="right">89.82</td>
<td align="right">75.97</td>
<td align="right">79.56</td>
<td></td>
<td></td>
</tr>
<tr>
<td><a href="https://github.com/allebee/jevk5">JevK5</a></td>
<td align="right">100.00</td>
<td align="right">97.22</td>
<td align="right">73.87</td>
<td align="right">64.50</td>
<td align="right">80.97</td>
<td align="right">89.13</td>
<td align="right">90.45</td>
<td align="right">85.16</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Intern-Decision-0.8B</td>
<td align="right">97.92</td>
<td align="right">80.56</td>
<td align="right">52.25</td>
<td align="right">77.35</td>
<td align="right">94.52</td>
<td align="right">88.61</td>
<td align="right">64.48</td>
<td align="right">79.38</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Intern-Decision-2B</td>
<td align="right">100.00</td>
<td align="right">84.72</td>
<td align="right">63.96</td>
<td align="right">79.35</td>
<td align="right">96.45</td>
<td align="right">89.96</td>
<td align="right">78.33</td>
<td align="right">84.68</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Intern-Decision-4B</td>
<td align="right">100.00</td>
<td align="right">98.61</td>
<td align="right">73.87</td>
<td align="right">80.55</td>
<td align="right">96.45</td>
<td align="right">90.82</td>
<td align="right">89.86</td>
<td align="right">90.02</td>
<td></td>
<td></td>
</tr>
<tr>
<td>Qwen/Qwen3.5-0.8B</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td align="right">40.37</td>
<td align="right">48.29</td>
</tr>
<tr>
<td>weihongliang/jevtrainer-qwen35-0.8b-2026-10-02</td>
<td align="right">100.00</td>
<td align="right">84.72</td>
<td align="right">50.45</td>
<td align="right">71.05</td>
<td align="right">95.81</td>
<td align="right">92.17</td>
<td align="right">83.44</td>
<td align="right"><b>82.52</b></td>
<td align="right">80.90</td>
<td align="right">66.71</td>
</tr>
</tbody>
</table>

Jev through Intern-Decision-4B are the seven-suite numbers published in the [Intern-Decision](https://github.com/internlm/Intern-Decision) README. That table does not include longdoc-dev or gui-v1.

Reproduce the seven suites with the checkpoint directory:

```bash
jt eval configs/eval/intern_accuracy_v1.yaml --set checkpoint=runs/jev-qwen35-0.8b
```

## Environment

Python 3.10 or newer. Install a CUDA build of PyTorch first, then this library. `pip install -e .` does not install the CUDA wheel.

| Package | Minimum |
| --- | --- |
| Python | 3.10 |
| torch, torchvision | 2.4, CUDA wheel from [pytorch.org](https://pytorch.org/get-started/locally/) |
| transformers | 5.5 |
| peft | 0.18 |
| accelerate | 1.10 |
| datasets | 4.0 |
| safetensors | 0.4 |
| pillow | 10 |
| numpy | 1.26 |
| pydantic | 2.7 |
| PyYAML | 6 |
| typer | 0.12 |

```bash
pip install torch torchvision
pip install "git+https://github.com/hongliang-wei/jevtrainer.git"
```

From a checkout, `pip install -e ".[serve]"` adds the HTTP server. This checkpoint is text and images. Audio and video packages are not required.

Inference is one forward pass in bf16. Training of this checkpoint used one RTX 3090 (48 GB).

## Download

```bash
huggingface-cli download weihongliang/jevtrainer-qwen35-0.8b-2026-10-02 --local-dir runs/jev-qwen35-0.8b
```

## Inference

You do not write the prompt by hand. Pass a state and typed questions. The `marker` readout turns them into one chat prompt and reads a probability for each option from a single forward pass.

Three question types:

| Type | You supply | The model returns |
| --- | --- | --- |
| `choice` | a dict of 2–255 option keys to descriptions | the chosen key, the full distribution, a confidence |
| `score` | an ordered list of 2–10 level descriptions | the expected level, the distribution, a confidence |
| `noul` | one yes/no proposition | P(yes) |

`calibration.json` supplies the temperature. The snippet below applies it.

```python
from jevtrainer.eval.runner import collect
from jevtrainer.model.load import load
from jevtrainer.predict import answer, probs, temperature_for
from jevtrainer.schema import Record

bundle = load("runs/jev-qwen35-0.8b", dtype="bf16", device="cuda")

record = Record.from_dict({
    "id": "demo",
    "state": "The cat sat on the mat.",
    "questions": {
        "topic": {
            "type": "choice",
            "instructions": "What is this sentence about?",
            "criteria": {
                "animal": "an animal is the subject",
                "weather": "the weather",
                "food": "something to eat",
            },
        },
        "clear": {
            "type": "noul",
            "instructions": "The sentence is grammatical.",
        },
        "quality": {
            "type": "score",
            "instructions": "How clear is the sentence?",
            "criteria": ["confusing", "acceptable", "clear"],
        },
    },
}).validate()

for row in collect(bundle, [record], batch_size=1):
    question = record.questions[row["name"]]
    temperature = temperature_for(bundle.temperature, question.type)
    print(row["name"], answer(question, probs(row["logits"], row["k"], temperature)))
```

An image is a path on `images`. Several questions in one record share one forward pass.

```python
record = Record.from_dict({
    "id": "screen",
    "state": "A settings window is open.",
    "images": ["screenshot.png"],
    "questions": {
        "control": {
            "type": "choice",
            "instructions": "Which control is shown?",
            "criteria": {"button": "a button", "slider": "a slider", "menu": "a menu"},
        }
    },
}).validate()
```

### Prompt the model actually reads

The readout builds this. Option symbols are `A`, `B`, `C`, ... in the order of `criteria`. The assistant message is a JSON skeleton with one `<decision>` placeholder per question. Logits at the token just before each placeholder, restricted to those symbols, are the distribution.

```text
System: You are a decision model. Read the state and the decision schema, then fill every field of the JSON object with the symbol of exactly one option.

User:
## State
The cat sat on the mat.

## Decision schema
### topic (choice)
What is this sentence about?
A = animal: an animal is the subject
B = weather: the weather
C = food: something to eat

### clear (noul)
The sentence is grammatical.
A = no
B = yes

### quality (score)
How clear is the sentence?
A = confusing
B = acceptable
C = clear

Assistant:
{
    "topic": "<decision>",
    "clear": "<decision>",
    "quality": "<decision>"
}
```

## HTTP

```bash
jt serve runs/jev-qwen35-0.8b
```

`POST /v1/systemone` takes the same `state` and `questions`. Default port is 8009.

```bash
curl -s http://127.0.0.1:8009/v1/systemone \
  -H "Content-Type: application/json" \
  -d "{\"state\":\"The cat sat on the mat.\",\"questions\":{\"topic\":{\"type\":\"choice\",\"instructions\":\"What is this sentence about?\",\"criteria\":{\"animal\":\"an animal is the subject\",\"weather\":\"the weather\"}}}}"
```

The response has `answers` (one entry per question), `usage.input_tokens`, and `latency_ms`. There are no generated tokens.
