# Draft plan: 9-page paper with LLM/VLM integration

**Working title:** *Does the Report Read the Heatmap? Localisation-Grounded Faithfulness of Template, LLM and VLM Summaries in a Deployed Chest Radiograph System*
**Venue:** NeurIPS 2026 Med-Reasoner workshop, double-blind, official `neurips_2026.sty`
**Length target:** 9 pages of body content (figures and tables count; references, checklist and appendices do not). Confirm the workshop's own page limit on OpenReview, because some workshops cap at 4 or 6 pages even when the style file allows 9.
**Builds on:** `paper/main.tex` (current 7-page version) and `paper/REVISION_NOTES.md` (supervisor comments 1 to 5)

---

## 1. The story in one paragraph

A deployed chest X-ray system shows a label, a heatmap and a written summary together, and the reader assumes the summary is based on the heatmap. We show that in LungLens it is not (the summary is a class-keyed template). We then plug in progressively stronger generators (a descriptor-gated template, a text LLM, and a vision-language model with and without the overlay) and progressively different heatmaps (Grad-CAM, Grad-CAM++, the amortised U-Net, a second backbone, a lung-masked map, and a permuted control). We measure, for every combination, how often the summary makes a location claim, how often that claim is right, and whether it survives the permuted-heatmap sanity check. The finding we expect: fluency rises from template to LLM to VLM, but grounding does not rise with it unless the generator is gated on measured evidence.

### Research questions

| RQ | Question | Answers supervisor comment |
| --- | --- | --- |
| RQ1 | What does the deployed summary actually condition on? | 1 |
| RQ2 | How does adding heatmap localisation change the summary, and is the change correct? | 2 |
| RQ3 | How much does the summary depend on *which* heatmap and *which* generator (LLM/VLM) is used? | 3 |
| RQ4 | What does grounding cost at serving time on a CPU-only deployment? | 5 |

Comment 4 (distillation) is answered in the System section by the "explanation amortisation" description, which is already written.

---

## 2. Page budget

| # | Section | Pages | Figures / tables in body |
| --- | --- | --- | --- |
| | Title, abstract | 0.5 | |
| 1 | Introduction | 1.0 | **Fig. 1** teaser (qualitative) |
| 2 | Related work | 0.75 | |
| 3 | System: LungLens and its localisation | 1.25 | **Fig. 2** system diagram |
| 4 | Report generators, including LLM and VLM | 1.0 | **Table 1** generator summary |
| 5 | Experimental design | 0.75 | **Table 2** evaluation grid |
| 6 | Results | 2.75 | **Fig. 3**, **Fig. 4**, **Fig. 5**, **Fig. 6**, **Table 3** |
| 7 | Discussion and limitations | 0.75 | |
| 8 | Broader impact and conclusion | 0.25 | |
| | **Total** | **9.0** | 6 figures, 3 tables |

Appendices (not counted): classifier detail, full prompts, template listing, extra qualitative grids, per-source breakdowns, latency per stage, human-rating protocol.

---

## 3. Section-by-section plan

### Abstract (0.25 to 0.5 page)
Use the structure agreed in `REVISION_NOTES.md` (broad field, problem, why it matters, "in this work", results, practical scope, code link). Update the results sentences once Section 6 numbers exist:
- template result (S0 never reads the mask);
- best grounded generator's claim precision vs. its permuted-heatmap precision;
- the VLM finding (with vs. without overlay);
- CPU latency of the grounded pipeline.

### 1. Introduction (1.0 page)
**Paragraphs**
1. Context: CXR triage in low-resource settings; deployed tools now bundle label + heatmap + text; VLMs make fluent text cheap.
2. The gap: each output is evaluated alone; nobody checks whether the text depends on the heatmap. Text-overlap metrics reward fluency.
3. What we do: audit a real deployed system, then swap in generators and heatmaps and measure grounding, with a permuted-heatmap sanity check.
4. Headline results (3 numbers).
5. Contributions list (4 items): audit; grounded generator family incl. LLM/VLM; heatmap x generator protocol with sanity check; CPU cost of grounding.

**Fig. 1 (teaser, qualitative, full width, ~0.35 page).** One radiograph, four columns: (a) input with the served overlay, (b) S0 template text, (c) LLM text (S2), (d) VLM text given the overlay (S4). Location claims highlighted green when they match the film's descriptors and red when they do not; a "no location claimed" chip where the gate fires. Pick a film where S4 confidently names a zone that the region does not support.
*Produce with:* new `make_fig_teaser.py` reading the evaluation CSV (Section 5).

### 2. Related work (0.75 page)
Three short paragraphs, most of which already exist in `main.tex`:
- **CXR classification and saliency faithfulness:** CheXNet, DenseNet, four-class work; Grad-CAM, Grad-CAM++; Adebayo et al. sanity checks; Kindermans et al.
- **Report generation and factuality:** R2Gen, Jing et al., factual-consistency metrics (Miura et al.), RadGraph-style entity metrics, region-grounded generation (RGRG).
- **Medical VLMs:** BioViL-T, LLaVA-Med, CheXagent, MedGemma, Med-PaLM M; hallucination and grounding evaluations of VLMs. **New:** add 4 to 6 citations here, verified against the papers.
End with one sentence positioning us: we study the *connection* between an explanation and the text next to it, in a deployed system, across generator types.

### 3. System: LungLens and its localisation (1.25 pages)
- **3.1 Data and classifier** (one paragraph, numbers only; full detail in Appendix A): eight sources, patient-grouped split, 96.18% accuracy, macro-F1 95.77%, reproducible split.
- **3.2 Localisation: explanation amortisation** (existing text, trimmed): frozen DenseNet teacher, Grad-CAM targets at `denseblock4.denselayer16.conv2`, U-Net student, no shared layers, Dice 0.22 with live Grad-CAM.
- **3.3 Region descriptors** (moved up from the design section): laterality, zone, extent, focality, in-lung fraction; the lung-field segmenter (`fig7_work/lungfield_unet.pth`, val Dice 0.98) used to compute them.

**Fig. 2 (system diagram, full width, ~0.45 page).** Update `figures/fig3.tex`: the report-generator block becomes a switch with four inputs (S0 template, S1 gated template, S2 LLM, S3/S4 VLM) and a new "grounding checker" block after it. Keep H-tags at the mask and S-tags at the generator.

### 4. Report generators (1.0 page)
Describe every generator the same way: inputs, what it can say, how it is gated. This section is the direct answer to supervisor comment 1.

| ID | Generator | Inputs | Location claims from | Gate |
| --- | --- | --- | --- | --- |
| S0 | Deployed template | class, confidence | none | confidence only |
| S1 | Descriptor template | + region descriptors | descriptors | in-lung >= 0.5 and region exists |
| S2 | Text LLM | class, confidence, descriptors (JSON) | descriptors, rephrased | same gate, enforced by schema |
| S3 | VLM, image only | radiograph, class, confidence | its own reading of the image | none |
| S4 | VLM, image + overlay | radiograph, overlay, class, confidence | image and overlay | none (optionally S4g: gated) |

**Table 1** = the table above, typeset. Full prompts go to Appendix C.
Key design point to state: S2 is *constrained* (structured JSON output, then rendered), so it can only rephrase evidence; S3/S4 are *free*, so they can add claims. That contrast is the experiment.

### 5. Experimental design (0.75 page)
**Films.** The 245 already-scored test films (102 pneumonia-as-Normal errors, 92 Shenzhen, 12 Montgomery, 40 stratified correct). Optionally add a random 200-film test sample so results are not dominated by hard cases; report both.

**Heatmap conditions**

| ID | Heatmap | Cost to produce | Status |
| --- | --- | --- | --- |
| H0 | none | none | done |
| H1 | live Grad-CAM (DenseNet-121) | CPU, forward + backward per film | run per film |
| H2 | Grad-CAM++ (DenseNet-121) | CPU, inference only | new |
| H3 | amortised U-Net (served) | CPU, one forward pass | done |
| H4 | Grad-CAM from a second backbone (ResNet-50 or EfficientNet-B0, same split) | one GPU training run | new |
| H5 | permuted: another film's map (derangement, seed 42) | none | done for S1 |
| H6 (optional) | H3 multiplied by the lung-field mask | CPU | new, a simple mitigation |

**Metrics** (all computed automatically from descriptors, no clinician needed):
- **Claim rate:** share of films where the summary states a location.
- **Grounding precision:** share of claims whose side/zone/extent match the film's descriptors under the *served* heatmap.
- **Pathology-plausible rate:** share of claims located inside the lung fields.
- **Phantom rate:** claims about an outline or region that does not exist.
- **Permutation drop:** precision under H_actual minus precision under H5. Large drop = the text reads the heatmap.
- **Label consistency (S3/S4):** does the text contradict the classifier's label?
- **Latency and cost** per summary on CPU (and API cost if hosted).
- **Optional human check:** 50 films, 1 to 2 raters (supervisor or a clinician if available) mark each claim correct / incorrect / unsupported; report agreement with the automatic metric.

**Claim extraction for free text (S3/S4).** A rule-based parser maps phrases to fields: side (`left|right|bilateral`), zone (`upper|mid|middle|lower|apical|basal`), extent (`focal|patchy|diffuse|extensive`). Validate the parser by hand on 40 outputs and report its accuracy in the appendix.

**Table 2 (evaluation grid)** = generators (rows) x heatmaps (columns), each cell marked run / not applicable. It replaces the current Table 1 in `main.tex`.

### 6. Results (2.75 pages)

**6.1 The deployed summary never reads the mask (0.25 page).** S0 is byte-identical under all H conditions; the 2 phantom-outline films. Text only.

**6.2 Main result: generator x heatmap (1.0 page).**
- **Table 3 (quantitative, main table):** rows S0 to S4 (and S4g), columns claim rate / grounding precision / permutation drop / phantom rate / label consistency / latency. One row per generator under H_actual, with H5 precision beside it.
- **Fig. 3 (quantitative, half width):** grouped bar chart, grounding precision per generator under H_actual vs. H5. The expected picture: S1 and S2 drop sharply under H5 (they read the heatmap); S3 barely changes (it never saw the heatmap); S4 somewhere in between.
- Text: this answers RQ2 and RQ3 directly, including "with vs. without heatmap" as S3 vs. S4 for the VLM.

**6.3 Which heatmap matters (0.75 page).**
- **Fig. 4 (quantitative, half width):** pairwise agreement matrix (median Dice) between H1, H2, H3, H4, H6 as a small heatmap. Shows how much the "evidence" depends on method and backbone.
- **Fig. 5 (qualitative, full width, ~0.4 page):** 4 films (rows) x heatmaps H1 / H2 / H3 / H4 (columns), each cell showing the overlay plus the S2 sentence it produced. Pick films where the side or zone flips between heatmaps.
- Text: summary changes driven by heatmap choice, and how often H4 (another model) agrees with H1 on the claimed side. This is supervisor comment 3.

**6.4 Why the gate fires, and where (0.5 page).**
- **Fig. 6 (quantitative, half width):** scatter of in-lung fraction vs. region area per film, coloured by claim outcome (correct / wrong / gated), with the 0.5 gate line; marker shape by source (Shenzhen, Montgomery, other). Shows that out-of-lung, diffuse maps are exactly the ones the gate removes, and that H6 moves films across the line.
- Text: link to the source-bias finding (Shenzhen diffuse, Montgomery empty).

**6.5 Serving cost (0.25 page).** Prose plus a line in Table 3: U-Net path 293 ms, Grad-CAM path 560 ms, descriptor extraction (measure), local LLM (measure), VLM (measure on CPU and GPU), hosted API round-trip (measure). State plainly which generators are feasible on the CPU-only laptop.

### 7. Discussion and limitations (0.75 page)
- Fluency vs. grounding: a better writer is not a more faithful one unless gated.
- Automatic descriptors as a proxy for truth: they measure agreement with the *heatmap*, not with pathology. Say this clearly; the human check (if done) partly addresses it.
- Dataset-source bias and label noise (existing text, shortened).
- Single deployed system; results may not transfer to others.
- VLM prompt sensitivity: report results for 2 prompt phrasings in the appendix.
- No clinical validation; the summary is not a finding.

### 8. Broader impact and conclusion (0.25 page)
Existing text, updated with one line on LLM/VLM risk: persuasive text with unsupported locations is the specific harm; gating and a visible "grounding check" badge are the mitigation.

---

## 4. Full figure and table list

| ID | Type | Placement | Content | Source / script | Status |
| --- | --- | --- | --- | --- | --- |
| Fig. 1 | Qualitative | Intro, full width | Teaser: one film, overlay + S0 / S2 / S4 text with claim highlighting | new `make_fig_teaser.py` | to do |
| Fig. 2 | Diagram | Sec. 3, full width | System + generator switch + grounding checker | `figures/fig3.tex` (edit) | update |
| Fig. 3 | Quantitative | Sec. 6.2, half | Grounding precision, H_actual vs H5, per generator | new `make_fig_grounding.py` | to do |
| Fig. 4 | Quantitative | Sec. 6.3, half | Heatmap agreement matrix (median Dice) | new, from per-film maps | to do |
| Fig. 5 | Qualitative | Sec. 6.3, full | 4 films x 4 heatmaps, with generated sentence | new `make_fig_heatmap_grid.py` | to do |
| Fig. 6 | Quantitative | Sec. 6.4, half | In-lung fraction vs area scatter, gate line | from `localisation_table.csv` | to do |
| Table 1 | Table | Sec. 4 | Generator definitions | LaTeX | to do |
| Table 2 | Table | Sec. 5 | Evaluation grid (run / n.a.) | LaTeX | update |
| Table 3 | Table | Sec. 6.2 | Main results per generator | evaluation CSV | to do |
| App. A | Tables + figure | Appendix | Per-class metrics, confusion matrix, per-source chart | existing | done |
| App. B | Figure | Appendix | Dice histogram U-Net vs Grad-CAM | `fig_dice_hist.pdf` | done |
| App. C | Listings | Appendix | S0 template verbatim; S2/S3/S4 prompts verbatim | text | partly done |
| App. D | Figure | Appendix | Existing six-panel qualitative overlays, descriptor distributions | existing | done |
| App. E | Table | Appendix | Latency per stage incl. LLM/VLM | extend `time_cpu.py` | update |
| App. F | Table | Appendix | Claim-parser validation, prompt-sensitivity, human-rating agreement | new | to do |

Chart style: one palette across all charts (the Okabe-Ito colours already used in `fig3.tex`), vector PDF output, same font size as the body.

---

## 5. LLM / VLM integration plan

There are two separate pieces of work. The **evaluation harness** is what the paper needs. The **app integration** is a demonstrator and is optional for the deadline.

### 5.1 Model choices

| Role | Primary (local, reproducible) | Alternative (hosted) | Notes |
| --- | --- | --- | --- |
| S2 text LLM | Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct, 4-bit, via Ollama or llama.cpp | Claude (`claude-haiku-4-5` for cost, `claude-sonnet-5` for quality) | Text-only input is tiny, so CPU inference is possible but slow; measure it |
| S3/S4 medical VLM | MedGemma-4B (multimodal) or CheXagent | | Medical-domain VLMs; run on the A10G for the evaluation |
| S3/S4 general VLM | Qwen2.5-VL-7B | Claude with image input | Lets us compare medical vs. general VLMs |

Pick **one local model per role as the main result** and at most one hosted model as a comparison. Check each model's licence and whether its terms allow medical research use, and cite the exact version and checkpoint hash. For review, describe hosted models generically if naming them could reveal identity (normally it does not).

**Data note:** all 245 films come from public datasets, so sending them to a hosted API is acceptable for the experiment. The app must **not** send user-uploaded radiographs to a hosted API by default (see 5.3).

### 5.2 Evaluation harness (needed for the paper)

New folder `report_eval/`, kept out of `app.py` (which is already 2,078 lines):

```
report_eval/
  descriptors.py     # region -> {side, zone, extent, focality, in_lung, has_region}
                     # (lift the logic already in fig7_work/extra_analysis.py and app.clean_region)
  generators.py      # s0_template, s1_gated, s2_llm, s3_vlm, s4_vlm, s4g_vlm_gated
  claim_parser.py    # free text -> {side, zone, extent} or None
  heatmaps.py        # H1..H6 producers; H4 loads the second backbone
  run_grid.py        # loops films x heatmaps x generators, caches outputs to JSONL
  score.py           # claim rate, precision, permutation drop, phantom, consistency
  tests/             # unit tests for descriptors, parser, gate, scoring
```

Pipeline per film and condition:
1. Produce the heatmap (Hk), threshold it with the app's own `clean_region` settings.
2. Compute descriptors against the lung-field mask.
3. Call the generator; for S2 request **structured JSON** that must match a schema, then render it to a sentence.
4. Parse claims from the text (S3/S4) or read them from the JSON (S2).
5. Score against the film's descriptors under the *served* heatmap; for H5, against the film's *own* descriptors.
6. Cache every raw output (prompt, response, model id, seed, latency) to JSONL so figures are reproducible without re-calling models.

**S2 prompt sketch (constrained):**
```
System: You write one-sentence location statements for a chest X-ray tool.
Use ONLY the fields provided. If "gate_pass" is false, set "location" to null.
Never add findings, sides or zones that are not in the input.
Return JSON: {"location": {"side": ..., "zone": ..., "extent": ..., "focality": ...} | null,
              "sentence": "..."}
User: {"predicted_class": "Covid-19", "confidence": 0.991, "gate_pass": true,
       "side": "left lung", "zone": "mid", "extent": "patchy", "focality": "multifocal"}
```
Then a deterministic check: every field in `location` must equal the input field, otherwise the output is replaced by the S1 sentence and counted as a **rejected LLM output** (report that rate too).

**S3 / S4 prompt sketch (free):**
```
S3: [radiograph] The classifier predicts {class} ({conf}%). In one or two sentences,
    describe where in the lungs the findings supporting this are located.
S4: [radiograph] [overlay image] Same text, plus: "The second image shows the
    classifier's heatmap; warmer colours mark the region that drove the prediction."
```
Temperature 0 (or fixed seed), max 80 tokens, same prompt for every film. Run a second phrasing for the prompt-sensitivity appendix.

**Compute estimate (to measure, not assumed):** 245 films x ~6 heatmaps x 5 generators is about 7,350 generator calls. Templates are instant; S2 is short text; S3/S4 are the expensive ones (245 x 2 image calls per heatmap condition that matters, H0 for S3 and H1/H3/H5 for S4). Run VLMs on the A10G instance used for training; time a 20-film pilot first and scale the grid if it is too slow.

### 5.3 App integration (optional demonstrator)

Add a **"Summary generator"** dropdown to the Gradio UI with: *Template (deployed)*, *Grounded template*, *LLM (grounded)*, *VLM (experimental)*.

Design rules:
- New module `report_llm.py`; `predict_image` calls one function `generate_report(mode, pred, mask, lung_mask, image)` and gets back `(text, claims, grounding_status)`.
- **Default mode stays local** (Grounded template). LLM/VLM modes use a local model if installed; a hosted API only if an API key is set in an environment variable and the user ticks an explicit "send image to external service" checkbox.
- Every LLM/VLM output passes through the same grounding checker as the harness. The UI shows a badge: *Grounded* (claims match the heatmap), *No location claimed* (gate fired), or *Unsupported claim removed*.
- Timeout (for example 20 s) and fallback to the grounded template on any error, with a visible note, never a silent failure.
- A fixed disclaimer under every summary: research tool, not a clinical finding.
- Log latency per stage so Appendix E can quote the app's own numbers.
- Unit tests for the checker and fallback paths; one end-to-end test that runs a sample image through each mode with the model mocked.

This gives the paper a real screenshot for the appendix (replacing the old Fig. 6 interface figure) showing the generator switch and the grounding badge, which also answers supervisor comment 1 visually.

---

## 6. Method extensions (system improvements)

The current 96.18% is inflated by source cues, so the goal is not a higher number on the same split. The goal is heatmaps that sit in the lungs, a localiser that genuinely shares layers with the classifier, and better accuracy on **sources the model never trained on**. Each extension below targets a weakness the existing analysis already measured.

### 6.1 Summary of extensions

| # | Extension | Weakness it targets (measured) | Paper role | Effort | Status |
| --- | --- | --- | --- | --- | --- |
| E1 | **Lung-constrained attention training** | 55% of served heatmaps sit mostly outside the lungs | New heatmap condition (H7) and a new classifier row | 1 GPU run per setting | **Prototype built** (`lung_attention/`); runs on AWS via `lung_attention/aws/` |
| E2 | **Shared-layer localiser** (decoder on frozen DenseNet features) | U-Net vs Grad-CAM median Dice 0.22; no shared layers (comment 4) | Replaces the amortised U-Net (H3') | New head + 1 localiser run | planned |
| E3 | **Leave-one-source-out (LOSO) evaluation** | Per-source accuracy 83.3% to 99.5%; source bias | External-validation table; judge E1/E5 on it | ~8 short runs | planned |
| E4 | **Self-verifying summary** (LLM/VLM + grounding checker + UI badge) | Summary never reads the mask (comment 1) | Generators S2 to S4, Section 5 of this plan | see Section 5 | planned |
| E5 | **Explanation consensus** (claim a location only where H1/H2/H3/H4 agree) | Heatmap methods disagree | Extra gated generator S1c; agreement map in the UI | small, after H1 to H4 exist | planned |
| E6 | **Similar-case retrieval** (nearest labelled training films in DenseNet embedding space) | Only one kind of explanation shown | App feature; appendix figure | small | optional |
| E7 | **Non-CXR input rejection** (out-of-distribution check) | App labels any image with high confidence | App safety feature; one appendix line | small | optional |
| T3a | **Label fix:** `Lung_Opacity` as its own class or dropped from Pneumonia | 82 of 102 pneumonia-as-Normal errors come from this folder | Classifier row; changes all classification numbers | 1 run | optional |
| T3b | **Source-adversarial head** (gradient reversal on the source label) | Source cues | Pairs with E1; judge on LOSO | 1 run per setting | optional |
| T3c | **Calibration + pneumonia-sensitive threshold + TTA** | Pneumonia recall 92.2% | App improvement; not a paper claim | small | optional |

Recommended core for the paper: **E1 + E2 + E4, all evaluated with E3.** That gives one chain: better heatmaps lead to more grounded summaries and better cross-source accuracy.

### 6.2 E1: lung-constrained attention training (prototype)

**Idea.** "Right for the right reasons" (Ross et al., 2017) and guided attention (GAIN, Li et al., 2018) applied to the DenseNet: add a loss that penalises the share of the class-activation map falling outside the lung field.

**Loss.** For each training image with ground-truth class *c*:
- CAM_c = ReLU(sum_k w_ck F_k), from the final DenseNet features F (1024 x 7 x 7) and the linear head weights w. For a global-average-pool plus linear head this is proportional to Grad-CAM at the final layer, so it is exact, cheap and needs no second-order gradients.
- Lung mask M from the existing lung-field U-Net (`fig7_work/lungfield_unet.pth`), computed **on the augmented batch** so flips, crops and rotations stay aligned; dilated by 7 px to keep pleural and apical evidence.
- L_attn = sum(CAM_c x (1 - M)) / sum(CAM_c), averaged over images with a plausible lung mask (10% to 70% of the frame).
- Total loss = weighted cross-entropy + lambda x L_attn, with lambda = 0 during warm-up epochs.

**Experiments.**
- lambda in {0, 0.5, 1, 2}. lambda = 0 through the same script is the control, so any difference is due to the loss alone.
- Select lambda on the **validation** split only; touch the test split once.
- Report: accuracy, macro-F1, per-class recall, per-source accuracy, and heatmap in-lung metrics measured with the **app's own served Grad-CAM** (layer `denseblock4.denselayer16.conv2`), not the CAM used in training.
- Then re-run the S1 grounding experiment with the E1 classifier as heatmap condition H7: does the gate fire less often, and does grounding precision under the permuted control still drop?

**Circularity caveat (must be stated in the paper).** The lung segmenter that supervises E1 is the same instrument that measures in-lung fraction. The evaluation therefore also scores against the **ground-truth lung masks** shipped with the COVID-19 Radiography Database, for every test film from that source, and reports those numbers separately.

**Paper placement.** One paragraph in System (Section 3), one row in Table 3, one bar in Fig. 3, and a before/after qualitative pair (same film, baseline vs E1 heatmap) added to Fig. 5.

### 6.3 E2: shared-layer localiser

Replace the separate `MultiTaskUNet` with a light decoder that takes the frozen DenseNet's feature maps from `denseblock1` to `denseblock4` as skip connections and predicts the Grad-CAM target. The classifier forward pass already computes those features, so the heatmap costs only the decoder. Report: Dice vs live Grad-CAM (baseline 0.22), parameter count (baseline 7.7 M), and CPU latency (baseline 293 ms). This also gives a correct answer to comment 4, because the student now genuinely shares the teacher's layers.

### 6.4 E3: leave-one-source-out

For each source with at least one disease class, train on the other seven and test on it. Report per-held-out-source accuracy and macro-F1 for the baseline, E1, and (optionally) T3b. This replaces "96.18% on a same-source split" with an external-validation claim, and it is the fairest test of whether E1 removes shortcuts.

---

## 7. Work plan

Durations are estimates.

| Step | Work | Est. time | Output |
| --- | --- | --- | --- |
| 1 | Build `report_eval/` descriptors, S0/S1, scoring; reproduce current numbers (92/245, 14/92) | 1 to 2 days | regression check passes |
| 2 | Per-film H1 and H2 on CPU; H6 lung-masked | 1 day | maps cached |
| 3 | Train second backbone (H4) on the same split on A10G | 1 day incl. eval | checkpoint + metrics |
| 4 | S2 LLM with JSON schema + checker; 20-film pilot, then full run | 1 to 2 days | JSONL + rejection rate |
| 5 | S3/S4 VLM pilot (20 films), claim parser + hand validation, full run | 2 to 3 days | JSONL + parser accuracy |
| 6 | Optional 50-film human rating | 1 day of rater time | agreement table |
| 7 | Figures 1, 3, 4, 5, 6 and Tables 1 to 3 | 2 days | PDFs in `paper/figures/` |
| 8 | Write sections 4 to 6, revise 1, 2, 7; update abstract | 3 days | 9-page draft |
| 9 | App integration (optional) + screenshot | 2 days | demo + appendix figure |
| 10 | Supervisor review, fact-check every number against the JSONL, compile, anonymise | 2 days | submission |
| E1 | Lung-constrained attention: lambda sweep on GPU, evaluate, re-run S1 grounding with H7 | 2 to 3 days | classifier rows + H7 |
| E2 | Shared-layer localiser: build, train, compare Dice/latency | 2 days | H3' condition |
| E3 | Leave-one-source-out for baseline and E1 | 2 days of GPU time | external-validation table |

**Minimum viable paper** if time is short: steps 1, 2, 4, 5 (one VLM only), 7, 8, 10, plus E1. H4, E2, E3, human rating and app integration can move to a journal version, and would be listed as open cells in Table 2.

---

## 8. Risks and how to handle them

| Risk | Mitigation |
| --- | --- |
| VLM too slow or too large | Run on the A10G; use a 4B model; reduce to H0/H3/H5 for S3/S4 |
| Claim parser misreads free text | Hand-validate 40 outputs, report parser accuracy, include a "no parseable claim" category |
| Descriptors are only a proxy for truth | Say so clearly; add the human check if a rater is available |
| Results look like "LLMs hallucinate", which is not new | Emphasise the permutation test and the gate: the contribution is a *measurement protocol* for grounding in a deployed pipeline, not the fact that VLMs err |
| Over the page limit | Move Fig. 6 and the H6 condition to the appendix first |
| Anonymity | No university name, GitHub user or screenshot branding in body or figures; anonymised code link |
| E1 is judged by the same lung segmenter that trains it | Also score against the Radiography Database's ground-truth lung masks; report both |
| E1 lowers accuracy | Expected to cost a little same-source accuracy; judge it on LOSO (E3) and heatmap quality, and report the trade-off honestly |
| Lung segmenter fails on paediatric or TB films it never saw | Validity gate (lung area 10% to 70%) skips the penalty on implausible masks; report how often it fires per source |
