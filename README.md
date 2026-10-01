# CONTACT

> [!IMPORTANT]
> **ALL samples will be released after paper publication.**

Code for **CONTACT: A Human-Grounded Benchmark and Surprisal-Based Predictive
Scorer for Conversational Naturalness**.

![CONTACT graphical abstract](assets/contact.png)

This release includes audio preprocessing, checkpoint inference, recording-level
scoring, and metric computation.
Model weights and audio are downloaded separately. The recording scores and
ratings used in the evaluation tables are included in `data/`.
The exact training inventory is included; a training launcher and baseline
inference pipelines are not bundled. Baseline protocols and released scores
are provided for evaluation.

## Installation

Use Python 3.10 and run commands from this directory.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

This installs the package with audio, inference and test dependencies.
Audio preprocessing requires a CUDA GPU for the affect teacher. If you only
need to reproduce tables from the included scores, use the smaller CPU install:

```bash
python -m pip install -c constraints.txt -e '.[test]'
```

## Reproduce the tables

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m contact.evaluate --data data --output outputs/evaluation
```

This reads the included recording scores; it does not run the neural models.
Outputs are `metrics.csv` (dev/test/all), `components.csv`, `table.tex`, and
`run.json`. The LaTeX table uses `booktabs` and `graphicx`.

To include paired session-bootstrap comparisons:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -m contact.evaluate --data data --output outputs/bootstrap \
  --bootstrap 20000 --seed 20260925
```

Bootstrap outputs contain paired differences, 95% confidence intervals and
unadjusted two-sided p-values. Comparisons use the strongest observed baseline
at each endpoint on the evaluation split.

## Score audio

Place downloaded audio in `datasets/`. See [datasets/README.md](datasets/README.md)
for the three CONTACT subsets, the Seamless Interaction data source, and
batch scoring commands. Scoring new audio does not require MOS labels.

### Weights

Download [contact.zip](https://github.com/review-anonymous-git/contact-code/releases/download/v1.0/contact.zip)
from the [v1.0 release](https://github.com/review-anonymous-git/contact-code/releases/tag/v1.0).
Extract the archive and place `contact.pt` at `checkpoints/contact.pt`, then
verify the extracted model:

```bash
python -m contact.checkpoint --checkpoint checkpoints/contact.pt
```

File sizes and SHA-256 checksums for the model and ZIP archive are listed in
[checkpoints/manifest.json](checkpoints/manifest.json).
Weights and generated caches are excluded from Git. The checkpoint contains
the predictor's inference parameters.

### Audio preparation

Install the affect-teacher wrapper at the following revision:

```bash
git clone https://github.com/tiantiaf0627/vox-profile-release.git third_party/vox-profile-release
git -C third_party/vox-profile-release checkout 85100e60844a3f324a139e24fb9225aa6d8e45d1
```

Use a stereo WAV with one speaker per channel, or two aligned mono tracks.
Do not duplicate a mixed signal into both channels.

```bash
python -m contact.prepare \
  --tracks /path/to/speaker_a.wav /path/to/speaker_b.wav \
  --recording-id example --output cache/example --device cuda:0

python -m contact.infer \
  --cache cache/example --corpus hai \
  --device cuda:0 --output outputs/example.csv
```

For stereo audio, replace `--tracks ... ...` with `--audio /path/to/stereo.wav`.
Corpus choices are `hh_turn`, `hh_emotion`, and `hai`; they select the stored
development-set normalization statistics. Scores are relative naturalness
scores (higher is better), **not calibrated 1–5 MOS predictions**.

Preparation downloads pinned Mimi and dimensional-affect checkpoints from
Hugging Face. Add `--local-files-only` and set `HF_HUB_OFFLINE=1` when all models
are already cached. Third-party models retain their own licenses.

For multiple prepared recordings, pass `--manifest recordings.csv` instead of
`--cache` and `--corpus`:

```csv
recording_id,corpus,cache_dir
example,hai,cache/example
```

Cache paths are relative to the manifest. Each cache contains two Mimi feature
arrays, VAD, affect targets and a checksum manifest. Preparation refuses to
overwrite an existing cache. Inference works on CPU as well, but is slower.

VAD, future-activity, silence and soft A/V targets are described in
[docs/preprocessing.md](docs/preprocessing.md).

If an inference CSV covers all 493 released recording IDs, evaluate it with
`python -m contact.evaluate --predictions outputs/predictions.csv --output outputs/rescored`.
This retains the released labels, split, baseline scores and dev normalization.

### Scoring

Inference uses 20-second windows with 8-second overlap, 3 seconds of left
context and a 2-second future guard. Overlapping scoring regions are assigned
to the first eligible window. The final checkpoint has separate timing and
affective Transformers and three readouts:

- **F:** negative mean future-activity NLL over valid frames.
- **S:** negative mean future joint-silence NLL. The target is the proportion
  of the next 4 seconds in which neither speaker talks, quantized into 10 bins.
- **A:** negative absolute difference between speakers' mean affective
  predictive gains. Gains compare the predicted likelihood with a training
  prior, averaged within each response and then across responses.

The affect head predicts separate eight-bin arousal and valence distributions
for each speaker. Targets come from response-aligned 3-second windows with a
1-second hop, at least 1.5 seconds of audio and 60% VAD activity. Both speakers
contribute to A, including in H–AI conversations. Future observations are used
as scoring targets, not as inputs to the causal predictor.

After subset-specific dev standardization:

```text
Timing  = 0.45 F + 0.55 S
Affect  = 0.25 A + 0.75 S
Overall = 0.50 Timing + 0.50 Affect
```

The formulas are in [contact/readout.py](contact/readout.py), the architecture
in [contact/model.py](contact/model.py), and the fixed settings in
[configs/scoring.json](configs/scoring.json). Affective and overall scores are
not reported for the H–H timing subset. Component ablations remove scores from
this same checkpoint; they are not separately retrained models.

### Development-set weight selection

To run the development-only grid search:

```bash
python -m contact.select_fusion --data data --output outputs/fusion_selection
```

This saves all candidate metrics, selected weights and fitted normalizers.
It does not change the fixed paper settings. See
[docs/fusion_selection.md](docs/fusion_selection.md) for the grid and objective.

## Training inventory

[training_manifest.csv](datasets/seamless_interaction/training_manifest.csv)
lists the exact 28,038 training recordings (1,635.831 dialogue hours, rounded
to 1,636 hours), with upstream IDs, paired audio paths and durations. Audio
download and preprocessing instructions are in
[datasets/seamless_interaction/README.md](datasets/seamless_interaction/README.md).

## Evaluation data

| Subset | Dev | Test | Total |
| --- | ---: | ---: | ---: |
| H–H turn-taking | 60 | 120 | 180 |
| H–H affective mismatch | 32 | 68 | 100 |
| H–AI | 70 | 143 | 213 |

[data/splits.csv](data/splits.csv) records sessions, pseudonymous speaker IDs
and conditions. Human speakers are disjoint across dev/test, including across
subsets. The evaluator checks IDs, checksums and speaker disjointness.

`ratings.csv` contains Participant (P), Supervisor (S), and Combined (C) MOS,
where `C = (P + S) / 2`. Rating definitions are recorded in
[data/release.json](data/release.json).

For H–H recordings, the file also includes:

- `participant_1_score`, `participant_2_score`: the two questionnaire ratings.
- `participant_1_role`, `participant_2_role`: instructed, uninstructed, or unresolved.
- `instructed_participant_score`, `uninstructed_participant_score`: ratings grouped by assignment role.
- `participant_role_status`: natural, unilateral, bilateral, or unresolved assignment.
- `uninstructed_mos_eligible`, `uninstructed_exclusion_reason`, `uninstructed_combined`:
  the eligibility and Combined target used in the additional analysis.

Participant numbers refer to questionnaire response order, not audio channels.
In natural conditions both participants are uninstructed, so their role-level
score is their mean. In Competitive Floor Conflict both participants receive
instructions, so no uninstructed score exists. This condition is excluded only
from the uninstructed-participant MOS analysis (10 dev and 20 test recordings);
the main results and discrimination metrics retain it. Equal questionnaire ratings can determine both
role-level scores even when the individual roles remain unresolved. Missing
role scores stay empty. These columns do not replace the existing P/S/C labels;
H–AI retains its dimension-specific ratings. Model predictions remain in
`scores.csv`, separate from human ratings.

Paired accuracy compares manipulated recordings with the natural recording
from the same session. C-index compares all natural–manipulated pairs within
the subset and split. Both give half credit for ties. MOS alignment uses
recording-level Spearman correlation.

Seven baseline score sets are included. Their recording-level adaptations are
described in [docs/baselines.md](docs/baselines.md); baseline inference pipelines
are not bundled. Audio examples are hosted separately at
[contact-samples](https://github.com/review-anonymous-git/contact-samples).

<!-- CONTACT_RESULTS_START -->
## Results

The table below includes the uninstructed-participant analysis. [LaTeX source](docs/results.tex) · [Standalone HTML](docs/results.html)

<details open>
<summary>H–H results</summary>

<table aria-label="H–H evaluation">
<caption>H–H evaluation</caption>
<thead>
<tr><th rowspan="4" scope="col">Scorer</th><th colspan="8">Discrimination</th><th colspan="12">MOS correlation</th></tr>
<tr><th colspan="4">Turn-taking</th><th colspan="4">Affective</th><th colspan="6">Turn-taking</th><th colspan="6">Affective</th></tr>
<tr><th colspan="2">Dev</th><th colspan="2">Test</th><th colspan="2">Dev</th><th colspan="2">Test</th><th colspan="3">Dev</th><th colspan="3">Test</th><th colspan="3">Dev</th><th colspan="3">Test</th></tr>
<tr><th scope="col">Acc.</th><th scope="col">C-I</th><th scope="col">Acc.</th><th scope="col">C-I</th><th scope="col">Acc.</th><th scope="col">C-I</th><th scope="col">Acc.</th><th scope="col">C-I</th><th scope="col">P</th><th scope="col">S</th><th scope="col">C</th><th scope="col">P</th><th scope="col">S</th><th scope="col">C</th><th scope="col">P</th><th scope="col">S</th><th scope="col">C</th><th scope="col">P</th><th scope="col">S</th><th scope="col">C</th></tr>
</thead>
<tbody>
<tr class="group"><th colspan="21">Baselines</th></tr>
<tr><th scope="row">UTMOSv2</th><td>44.0</td><td>.47</td><td>50.0</td><td>.50</td><td>54.2</td><td>.51</td><td>51.0</td><td>.46</td><td>.03</td><td>.04</td><td>.04</td><td>-.01</td><td>-.02</td><td>-.01</td><td>.06</td><td>-.05</td><td>.00</td><td>-.21</td><td>-.07</td><td>-.16</td></tr>
<tr><th scope="row">VAP</th><td>70.0</td><td>.70</td><td>80.0</td><td>.76</td><td>45.8</td><td>.48</td><td>41.2</td><td>.47</td><td>.19</td><td>.12</td><td>.17</td><td>.12</td><td>.16</td><td>.15</td><td>.08</td><td>-.22</td><td>-.08</td><td>-.27</td><td>-.19</td><td>-.27</td></tr>
<tr><th scope="row">DualTurn</th><td><strong>84.0</strong></td><td><strong>.80</strong></td><td>85.0</td><td>.77</td><td>58.3</td><td>.59</td><td>60.8</td><td>.55</td><td>.27</td><td>.24</td><td>.28</td><td>.18</td><td>.17</td><td>.19</td><td>.15</td><td>.02</td><td>.06</td><td>.03</td><td>-.03</td><td>.00</td></tr>
<tr><th scope="row">Talking Turns</th><td>68.0</td><td>.65</td><td>72.0</td><td>.67</td><td>45.8</td><td>.49</td><td>54.9</td><td>.58</td><td>-.05</td><td>-.02</td><td>-.04</td><td>.06</td><td>.03</td><td>.06</td><td>.13</td><td>.17</td><td>.19</td><td>.04</td><td>.00</td><td>.03</td></tr>
<tr><th scope="row">UniSRM</th><td>68.0</td><td>.65</td><td>63.0</td><td>.61</td><td>62.5</td><td>.56</td><td>42.2</td><td>.43</td><td>.27</td><td>.19</td><td>.26</td><td>.02</td><td>.03</td><td>.03</td><td>-.06</td><td>.10</td><td>-.01</td><td>-.22</td><td>-.13</td><td>-.21</td></tr>
<tr><th scope="row">TRACE</th><td>64.0</td><td>.56</td><td>53.0</td><td>.51</td><td>50.0</td><td>.49</td><td>62.7</td><td>.61</td><td>.17</td><td>.24</td><td>.21</td><td>.00</td><td>.10</td><td>.06</td><td>-.06</td><td>.12</td><td>.06</td><td>.17</td><td>.15</td><td>.17</td></tr>
<tr><th scope="row">TimingStats</th><td>62.0</td><td>.64</td><td>61.0</td><td>.61</td><td>50.0</td><td>.50</td><td>56.9</td><td>.52</td><td>.08</td><td>.22</td><td>.16</td><td>.00</td><td>.18</td><td>.09</td><td>-.26</td><td>.00</td><td>-.10</td><td>-.07</td><td>.10</td><td>.01</td></tr>
<tr class="group"><th colspan="21">Ours and score-component ablations</th></tr>
<tr><th scope="row">Ours</th><td>78.0</td><td>.72</td><td><strong>86.0</strong></td><td><strong>.80</strong></td><td><strong>83.3</strong></td><td><strong>.77</strong></td><td><strong>82.4</strong><sup>†</sup></td><td><strong>.72</strong></td><td><strong>.28</strong></td><td><strong>.37</strong></td><td><strong>.35</strong></td><td><strong>.35</strong></td><td><strong>.33</strong></td><td><strong>.36</strong><sup>†</sup></td><td><strong>.39</strong></td><td><strong>.33</strong></td><td><strong>.40</strong></td><td><strong>.26</strong></td><td><strong>.29</strong></td><td><strong>.28</strong></td></tr>
<tr><th scope="row">w/o A–V</th><td>78.0</td><td>.72</td><td><strong>86.0</strong></td><td><strong>.80</strong></td><td><strong>83.3</strong></td><td>.75</td><td>74.5</td><td>.71</td><td><strong>.28</strong></td><td><strong>.37</strong></td><td><strong>.35</strong></td><td><strong>.35</strong></td><td><strong>.33</strong></td><td><strong>.36</strong></td><td>.38</td><td>.31</td><td>.39</td><td>.24</td><td><strong>.29</strong></td><td>.27</td></tr>
<tr><th scope="row">w/o future joint silence</th><td>80.0</td><td>.78</td><td>84.0</td><td>.77</td><td>62.5</td><td>.62</td><td>60.8</td><td>.64</td><td><strong>.28</strong></td><td>.28</td><td>.31</td><td>.20</td><td>.19<sup>‡</sup></td><td>.22</td><td>.13</td><td>.12</td><td>.15</td><td>.07</td><td>.17</td><td>.12</td></tr>
<tr><th scope="row">w/o future voice activity</th><td>60.0</td><td>.58</td><td>71.0<sup>‡</sup></td><td>.69<sup>‡</sup></td><td><strong>83.3</strong></td><td><strong>.77</strong></td><td><strong>82.4</strong></td><td><strong>.72</strong></td><td>.26</td><td>.32</td><td>.31</td><td>.33</td><td>.31</td><td>.34</td><td><strong>.39</strong></td><td><strong>.33</strong></td><td><strong>.40</strong></td><td><strong>.26</strong></td><td><strong>.29</strong></td><td><strong>.28</strong></td></tr>
<tr class="group"><th colspan="21">Uninstructed-participant sensitivity</th></tr>
<tr><th scope="row">Ours</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>.40</td><td>.44</td><td>.45</td><td><strong>.44</strong><sup>†</sup></td><td>.44<sup>†</sup></td><td><strong>.48</strong><sup>†</sup></td><td><strong>.23</strong></td><td><strong>.33</strong></td><td><strong>.35</strong></td><td><strong>.23</strong></td><td><strong>.29</strong></td><td><strong>.28</strong></td></tr>
<tr><th scope="row">w/o A–V</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>.40</td><td>.44</td><td>.45</td><td><strong>.44</strong></td><td>.44</td><td><strong>.48</strong></td><td>.21</td><td>.31</td><td>.31</td><td>.20</td><td><strong>.29</strong></td><td>.26</td></tr>
<tr><th scope="row">w/o future joint silence</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>.31</td><td>.28</td><td>.32</td><td>.29</td><td>.25<sup>‡</sup></td><td>.30<sup>‡</sup></td><td>.22</td><td>.12</td><td>.23</td><td>.10</td><td>.17</td><td>.17</td></tr>
<tr><th scope="row">w/o future voice activity</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td><strong>.41</strong></td><td><strong>.51</strong></td><td><strong>.50</strong></td><td>.42</td><td><strong>.47</strong></td><td><strong>.48</strong></td><td><strong>.23</strong></td><td><strong>.33</strong></td><td><strong>.35</strong></td><td><strong>.23</strong></td><td><strong>.29</strong></td><td><strong>.28</strong></td></tr>
<tr class="group"><th colspan="21">Reference (not an audio scorer)</th></tr>
<tr><th scope="row">Inter-rater ρ (P–S)</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>.75</td><td>—</td><td>—</td><td>.76</td><td>—</td><td>—</td><td>.66</td><td>—</td><td>—</td><td>.69</td></tr>
</tbody>
</table>

</details>

<details open>
<summary>H–AI results</summary>

<table aria-label="H–AI evaluation">
<caption>H–AI evaluation</caption>
<thead>
<tr><th rowspan="3" scope="col">Scorer</th><th colspan="6">Turn-taking</th><th colspan="6">Affective</th><th colspan="6">Overall</th></tr>
<tr><th colspan="3">Dev</th><th colspan="3">Test</th><th colspan="3">Dev</th><th colspan="3">Test</th><th colspan="3">Dev</th><th colspan="3">Test</th></tr>
<tr><th scope="col">P</th><th scope="col">S</th><th scope="col">C</th><th scope="col">P</th><th scope="col">S</th><th scope="col">C</th><th scope="col">P</th><th scope="col">S</th><th scope="col">C</th><th scope="col">P</th><th scope="col">S</th><th scope="col">C</th><th scope="col">P</th><th scope="col">S</th><th scope="col">C</th><th scope="col">P</th><th scope="col">S</th><th scope="col">C</th></tr>
</thead>
<tbody>
<tr class="group"><th colspan="19">Baselines</th></tr>
<tr><th scope="row">UTMOSv2</th><td>.12</td><td>.22</td><td>.20</td><td>.07</td><td>-.02</td><td>.02</td><td>.15</td><td>.10</td><td>.15</td><td>.08</td><td>-.09</td><td>-.01</td><td>.21</td><td>.12</td><td>.19</td><td>.06</td><td>-.06</td><td>-.01</td></tr>
<tr><th scope="row">VAP</th><td>.20</td><td>.20</td><td>.24</td><td>.23</td><td>.19</td><td>.23</td><td>.09</td><td>.09</td><td>.09</td><td>.21</td><td>.19</td><td>.22</td><td>.21</td><td>.12</td><td>.19</td><td>.20</td><td>.18</td><td>.21</td></tr>
<tr><th scope="row">DualTurn</th><td>.46</td><td><strong>.41</strong></td><td>.50</td><td>.36</td><td>.27</td><td>.35</td><td>.38</td><td>.22</td><td>.33</td><td>.29</td><td>.17</td><td>.26</td><td>.34</td><td>.30</td><td>.37</td><td>.30</td><td>.15</td><td>.26</td></tr>
<tr><th scope="row">Talking Turns</th><td>.05</td><td>.06</td><td>.08</td><td>.21</td><td>.09</td><td>.18</td><td>.06</td><td>-.13</td><td>-.06</td><td>.18</td><td>.04</td><td>.13</td><td>.05</td><td>-.02</td><td>.04</td><td>.18</td><td>.13</td><td>.19</td></tr>
<tr><th scope="row">UniSRM</th><td>.25</td><td>.29</td><td>.30</td><td>.18</td><td>.14</td><td>.18</td><td>.18</td><td>.16</td><td>.15</td><td>.19</td><td>.23</td><td>.24</td><td>.21</td><td>.16</td><td>.20</td><td>.18</td><td>.23</td><td>.22</td></tr>
<tr><th scope="row">TRACE</th><td>.10</td><td>.09</td><td>.09</td><td>.11</td><td>.09</td><td>.10</td><td>.08</td><td>.24</td><td>.17</td><td>.10</td><td>.04</td><td>.06</td><td>.16</td><td>.21</td><td>.19</td><td>.09</td><td>.14</td><td>.13</td></tr>
<tr><th scope="row">TimingStats</th><td>.02</td><td>-.07</td><td>-.05</td><td>-.08</td><td>.09</td><td>.01</td><td>.06</td><td>.13</td><td>.09</td><td>-.05</td><td>.18</td><td>.07</td><td>.09</td><td>.14</td><td>.09</td><td>.03</td><td>.18</td><td>.09</td></tr>
<tr class="group"><th colspan="19">Ours and score-component ablations</th></tr>
<tr><th scope="row">Ours</th><td>.52</td><td>.39</td><td>.52</td><td>.40</td><td><strong>.31</strong></td><td><strong>.40</strong></td><td><strong>.49</strong></td><td><strong>.27</strong></td><td><strong>.41</strong></td><td><strong>.36</strong></td><td>.29</td><td><strong>.37</strong></td><td>.45</td><td>.38</td><td>.47</td><td>.41<sup>†</sup></td><td>.23</td><td>.36<sup>†</sup></td></tr>
<tr><th scope="row">w/o A–V</th><td>.52</td><td>.39</td><td>.52</td><td>.40</td><td><strong>.31</strong></td><td><strong>.40</strong></td><td>.47</td><td>.26</td><td>.39</td><td>.34</td><td>.23<sup>‡</sup></td><td>.32<sup>‡</sup></td><td>.44</td><td>.38</td><td>.47</td><td>.40</td><td>.21</td><td>.34</td></tr>
<tr><th scope="row">w/o future joint silence</th><td>.42</td><td>.37</td><td>.46</td><td>.33</td><td>.26</td><td>.33<sup>‡</sup></td><td>.27</td><td>.07</td><td>.22</td><td>.22</td><td><strong>.32</strong></td><td>.33</td><td>.37</td><td>.26</td><td>.36</td><td>.35</td><td>.23</td><td>.33</td></tr>
<tr><th scope="row">w/o future voice activity</th><td><strong>.55</strong></td><td>.39</td><td><strong>.53</strong></td><td><strong>.41</strong></td><td><strong>.31</strong></td><td><strong>.40</strong></td><td><strong>.49</strong></td><td><strong>.27</strong></td><td><strong>.41</strong></td><td><strong>.36</strong></td><td>.29</td><td><strong>.37</strong></td><td><strong>.46</strong></td><td><strong>.40</strong></td><td><strong>.49</strong></td><td><strong>.43</strong></td><td>.22</td><td><strong>.37</strong></td></tr>
<tr><th scope="row">w/o timing outputs</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>.29</td><td>.10</td><td>.22</td><td>.25</td><td><strong>.25</strong></td><td>.27</td></tr>
<tr class="group"><th colspan="19">Reference (not an audio scorer)</th></tr>
<tr><th scope="row">Inter-rater ρ (P–S)</th><td>—</td><td>—</td><td>.58</td><td>—</td><td>—</td><td>.55</td><td>—</td><td>—</td><td>.38</td><td>—</td><td>—</td><td>.39</td><td>—</td><td>—</td><td>.55</td><td>—</td><td>—</td><td>.54</td></tr>
</tbody>
</table>

</details>

<p>Acc.: paired accuracy (%); C-I: C-index; P/S/C: Spearman correlation with Participant,
Supervisor and Combined ratings. <sup>†</sup> marks Ours significantly above the highest-scoring
baseline at that endpoint. <sup>‡</sup> marks an ablation significantly below Ours in the targeted
component comparisons. Tests use 20,000 paired session-bootstrap resamples and unadjusted two-sided p &lt; .05.</p>
<p>Main H–H rows use the two-participant mean for P. In the uninstructed-participant block,
P is the uninstructed participant's rating for unilateral manipulations and the two-participant mean
for natural conditions. Competitive Floor Conflict is excluded from H–H MOS in this block
(10 dev / 20 test recordings): both participants receive instructions, so neither is uninstructed.
The main results retain Competitive Floor Conflict. The same eligible recordings are used for
P/S/C: turn-taking 50 dev / 100 test; affective 32 dev / 68 test. Discrimination retains all recordings.
Dashes in the sensitivity block's discrimination columns indicate unchanged results.
Unchanged H–AI results are not repeated. Reference rows retain the main rating protocol.</p>
<p>Bold follows the supplied table, with maxima shown separately within the H–H sensitivity block.
Component ablations remove scores from the same checkpoint.</p>
<!-- CONTACT_RESULTS_END -->

## Tests

```bash
python -m pytest -q
```

Tests requiring PyTorch or Transformers are skipped when those optional
dependencies are absent. No test downloads a model or requires a GPU.
