# Baseline scoring

These are fixed-weight pretrained models, except for the TimingStats reference
distribution fitted on natural training recordings. The scores in `data/scores.csv`
are recording-level outputs. Baseline inference pipelines are not bundled.

| Scorer | Model inputs and recording-level readout |
| --- | --- |
| VAP | Separate 16-kHz channels; 50-Hz predictions; 256 future-activity states; non-overlapping 60-s chunks. Negative mean categorical NLL over valid frames, excluding the first 3 s of the recording and incomplete chunk-end targets. |
| DualTurn | Separate 24-kHz channels; 12.5-Hz predictions; eight native future-activity outputs. Soft-occupancy BCE summed across outputs, then negative mean over valid frames. Non-overlapping 30-s chunks; exclude the first 3 s and incomplete recording-end targets. |
| Talking Turns | Whisper-medium event predictor; mono mixture at 16 kHz; predictions every 40 ms using at most 30 s of history. Automatic channel-wise pyannote VAD and faster-whisper-medium references. Equal mean of the ten directional event accuracies with available events. Pool H–H speaker counts; use the AI as focal speaker in H–AI. |
| UTMOSv2 | Five pretrained folds, five stochastic crop repetitions per fold; 16-kHz mono with silence retained. Average all 25 predictions. Not exhaustive sliding-window coverage. |
| UniSRM | Task-4 rubric adapted to one complete observed response and preceding audio history. Greedy decoding, normally at most 2,048 new tokens. H–H: mean within each speaker, then equal speaker mean. H–AI: eligible AI responses. Emotion & Prosody Match supplies affect; Overall Naturalness supplies overall and the timing proxy. |
| TRACE | Speech-only naturalness logit. Channel-wise 3-s windows with 1-s hop; Whisper-large-v3 emotion embeddings; 1,280 dimensions, L2-normalized and interleaved. At most 256 windows per speaker. |
| TimingStats | Negative squared Mahalanobis distance of eight standardized VAD statistics. Ledoit–Wolf covariance fitted on 4,263 natural training-reference recordings, without CONTACT ratings. |

Talking Turns retains the author thresholds: `p_T-p_C > 0` for turn change and
yielding, `p_BC > 0.1` for backchanneling, `p_I-p_C > -0.45` for interruption,
and `p_T-p_C > -0.1` for interruption acceptance. Undefined event metrics are
omitted, not assigned zero. The event predictor and thresholds are author-native;
their aggregation into a recording-level naturalness score is our adaptation.

UniSRM responses merge same-speaker VAD segments across gaps up to 0.5 s unless
the other speaker begins in the gap. Responses must last at least 0.5 s and have
at least 0.5 s of preceding speech from each speaker. History runs from recording
start to response onset. Only the evaluated speaker's channel is used for the
response. Audio truncation is disabled. Scores use no fitted rubric weights,
duration weighting or tail pooling.

TimingStats features are mean/std of signed floor-transfer offsets and speech-run
durations, overlap ratio, joint-silence ratio, speaker-switch rate and short-segment
rate. Same-speaker gaps up to 0.2 s are merged when the partner is inactive;
short segments last at most 0.8 s.
