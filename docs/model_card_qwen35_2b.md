---
license: apache-2.0
base_model: Qwen/Qwen3.5-2B
tags:
- jev
- marker
- qwen3.5
- lora
---

# jevtrainer-qwen35-2b-2026-10-05

<p align="center"><a href="https://github.com/hongliang-wei/jevtrainer"><img alt="GitHub: hongliang-wei/jevtrainer" src="https://img.shields.io/badge/GitHub-hongliang--wei%2Fjevtrainer-181717?logo=github&logoColor=white"></a></p>

A Jev-style decision model. One forward pass returns a probability for every option. It does not generate text. The readout is `marker`. The base `Qwen/Qwen3.5-2B` is LoRA fine-tuned (r=64, alpha=128), with the vision tower frozen. Training finished on **2026-10-05**.

The training config is [`configs/repro/qwen35_2b_lora_v4.yaml`](https://github.com/hongliang-wei/jevtrainer/blob/main/configs/repro/qwen35_2b_lora_v4.yaml) in [jevtrainer](https://github.com/hongliang-wei/jevtrainer): the same public v4 mixture as the 0.8B run, about 540k training records, one epoch. Holdout is 11,706 records, accuracy 88.19%.

This repo is the LoRA adapter plus the readout. `adapter/` holds the PEFT weights. The same directory has `readout.safetensors`, `readout.json`, `calibration.json`, and `tokenizer/`. `load()` reads those together and downloads `Qwen/Qwen3.5-2B` for the frozen base. Pass a local directory; a Hub repo id is not accepted directly.

## Evaluation

Accuracy in percent.

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
<td><a href="https://huggingface.co/weihongliang/jevtrainer-qwen35-0.8b-2026-10-02">jevtrainer-0.8b</a></td>
<td align="right">100.00</td>
<td align="right">84.72</td>
<td align="right">50.45</td>
<td align="right">71.05</td>
<td align="right">95.81</td>
<td align="right">92.17</td>
<td align="right">83.44</td>
<td align="right">82.52</td>
<td align="right">80.90</td>
<td align="right">66.71</td>
</tr>
<tr>
<td><a href="https://huggingface.co/weihongliang/jevtrainer-qwen35-2b-2026-10-05">jevtrainer-2b</a></td>
<td align="right">100.00</td>
<td align="right">90.28</td>
<td align="right">52.25</td>
<td align="right">72.50</td>
<td align="right">96.13</td>
<td align="right">92.64</td>
<td align="right">92.67</td>
<td align="right"><b>85.21</b></td>
<td align="right">82.93</td>
<td align="right">69.08</td>
</tr>
</tbody>
</table>

```bash
huggingface-cli download weihongliang/jevtrainer-qwen35-2b-2026-10-05 --local-dir runs/jev-qwen35-2b
jt eval configs/eval/intern_accuracy_v1.yaml --set checkpoint=runs/jev-qwen35-2b
```
