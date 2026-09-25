# Revision notes: response to Dr Imad Gohar's comments

**Paper:** LungLens, reframed as *The Summary Never Reads the Mask: Localisation-Grounded Report Faithfulness in a Deployed Chest Radiograph System*
**Target:** NeurIPS 2026 Workshop on Medical Reasoning with Vision Language Foundation Models (Med-Reasoner), double-blind workshop track, official `neurips_2026.sty`
**Deadline:** extended to 5 September 2026 (per the organisers' email to Dr Imad)
**Compared against:** the report version `LungLens Evan Lo Jen Zhen.docx` (figure numbers below refer to that version unless stated)

---

## Summary of the reframing

The report was written as a classification paper: DenseNet-121 plus Grad-CAM, a confusion matrix, per-class metrics, and the summary mentioned in one line. As Dr Imad pointed out, DenseNet-121 with Grad-CAM is well-trodden ground, so a classification framing reads as old work at any venue looking for new results.

The paper now asks a different question: **does the natural-language summary actually depend on the localisation shown next to it?** The classifier is treated as a fixed upstream component and its full evaluation moves to Appendix A. The main text now covers:

1. an audit of what the deployed summary generator reads;
2. a comparison of summaries with and without heatmap localisation;
3. a sanity check that swaps each film's heatmap for another film's heatmap;
4. an evaluation grid that crosses heatmap source with generator type, which is the experiment design Dr Imad described.

This framing fits Med-Reasoner, because the workshop is about whether vision-language outputs reason from the image.

---

## Major comments

### Comment 1: The Summary in Fig. 3 and Fig. 6 is not connected to the rest of the system; how it was produced is not stated

**Problem in the report.** Section III-F had one sentence ("A short clinical-language description is generated from the prediction"). Fig. 3 showed a "Summary" output, and Fig. 6 showed summary text, but the paper never said what produced it.

**What we found when we checked the code.** The summary is a static Python dictionary with four entries, keyed on the predicted class. It is joined to one of five fixed sentences, chosen by three flags (below the 60% confidence threshold, confident non-Normal finding, mask available). No coordinate, area, side, zone or component count from the heatmap is ever read. The heatmap and the summary are disconnected.

**Fixes made.**
- New Section 3.2, *Report generation*, states exactly what the generator is, what its inputs are, the confidence gate, and one failure it causes (it can say "the outline marks the most influential area" on a film where no outline was drawn; this happens on 2 of 245 films).
- Appendix B reproduces the generator verbatim (`app.py`, `predict_image`), so reviewers can check the claim directly.
- Fig. 3 (system overview) now draws the report generator as its own block, shows which inputs reach it (label and confidence only for the deployed template), and tags where each heatmap condition (H0 to H5) and generator condition (S0 to S4) enters.
- This finding is now the paper's central result (Section 5.1) instead of a detail.

**Still to do.**
- [ ] Fig. 6 (the interface screenshot) was dropped from the reframed paper. Either restore it with the summary panel labelled "S0: class-keyed template", or replace it with the paired S0 vs S1 figure below, so the reader sees where the text in the UI comes from.

### Comment 2: How does the heatmap localisation help refine the Summary? How do the summaries with and without localisation compare?

**Problem in the report.** The report claimed the overlay and summary work together, but never compared a summary built with localisation against one built without it.

**Fixes made.** Sections 4 and 5 now run that comparison on the 245 test films already scored in the failure analysis:

| Condition | Meaning |
| --- | --- |
| **S0** | Deployed template (class and confidence only) |
| **S1** | New descriptor-gated template: states side, vertical zone, extent and focality read from the heatmap region, and **says nothing about location** when the region is mostly outside the lung (in-lung fraction below 0.5) or no region survives the threshold |
| **H0** | No heatmap |
| **H_actual** | The heatmap the system actually served (U-Net mask, or live Grad-CAM fallback) |
| **H5** | Degenerate control: each film gets a *different* film's heatmap (fixed derangement, seed 42) |

Results, all without retraining:
- **Without localisation (S0, or S1 under H0)** the summary can only restate the class. S0 prints identical text under every heatmap condition.
- **With localisation (S1 under H_actual)** the summary makes a grounded spatial claim on 92 of 245 films (37.6%), for example "left lung, mid zone, patchy, multifocal". On the other 153 (62.4%) the gate stops it, because either no region survives (41 films) or the region sits mostly outside the lung fields.
- **With the wrong film's heatmap (S1 under H5)** the claim rate stays at 92/245, but only 14 of 92 claims (15.2%) match the film's real side, zone and extent. So when the claims are correct under H_actual, that is because the heatmap is being read, not by chance.

So the localisation helps the summary in one specific way: it adds a *where* to the *what*. It also limits the summary, because the claim can only be as accurate as the heatmap under it. The U-Net mask and the live Grad-CAM map for the same film agree at a median Dice of only 0.22.

**Still to do.**
- [ ] Commit `figures/paired_summary.pdf` (S0 and S1 text on the same film, side by side). `main.tex` currently shows a typeset fallback box.
- [ ] Report the S1 x H0 cell as a number in Section 5.2 (claim rate 0% by construction). Table 1 marks it "measured" but the text never states it.
- [ ] Split H_actual into H1 (Grad-CAM) and H3 (U-Net) per film by computing both maps for all 245 films. This is CPU-feasible and would show directly how the choice of heatmap changes the summary.

### Comment 3: The paper is focused on classification with an old method; if the focus is the Summary, the experiments should vary heatmap models and VLMs

**Problem in the report.** Most of the report was about the image model (confusion matrix, per-class metrics, per-source probe). DenseNet-121 plus Grad-CAM is not new, so a reviewer at a state-of-the-art venue would see a well-known result.

**Fixes made.**
- The title, abstract, introduction and contributions now target report faithfulness, not classification accuracy.
- The classifier is summarised in one paragraph (Section 5). Its table, confusion matrix and per-source chart are in Appendix A.
- Section 4 defines the grid Dr Imad described: **heatmap source** (H0 none, H1 Grad-CAM, H2 Grad-CAM++, H3 U-Net, H4 Grad-CAM from a second backbone, H5 permuted) crossed with **generator** (S0 deployed template, S1 descriptor template, S2 text-only LLM, S3 VLM given the image, S4 VLM given the image plus overlay).
- Related work now covers report generation and faithfulness (R2Gen, factual-consistency metrics, region-grounded generation, BioViL-T) and saliency sanity checks (Adebayo et al.), not only classification.

**Still to do (the most important remaining gap).** Only the S0 and S1 rows are filled. For a workshop about *vision language foundation models*, a paper with no VLM result is exposed. In order of value per effort, we should:
- [ ] **S3/S4, at least one VLM** (for example an open medical VLM, or a general VLM through an API) generating the summary from the image alone (S3) and from the image plus overlay (S4), scored with the same grounding-precision and H5 metrics. This is the result reviewers at this venue will look for first.
- [ ] **S2, a text-only LLM** given class, confidence and the S1 descriptors. This is cheap and shows whether fluency adds claims the evidence does not support.
- [ ] **H4, a second backbone** (for example ResNet-50 or EfficientNet-B0) trained on the same split. This separates "the summary tracks the pathology" from "the summary tracks this one model". It needs one GPU training run of the kind already done on the A10G.
- [ ] **H2, Grad-CAM++**, inference only.
- [ ] If an LLM is used as a judge, report its agreement with human labels on a subsample.

---

## Minor comments

### Comment 4: The teacher-student distillation is not explained; which layers are shared?

**Problem in the report.** Section III-E called the U-Net stage "knowledge distillation" but never said which layers, logits or features connect the two networks. Dr Imad was right to doubt it: **no layers are shared.** The only link is the Grad-CAM map, which is not a learned layer.

**Fixes made.**
- Section 3.1 is renamed *Localisation: explanation amortisation, not knowledge distillation* and states plainly that nothing is shared between the two networks: no layers, no logit or feature matching, and no gradient path from student to teacher.
- It now gives the full procedure: the frozen DenseNet-121 produces a Grad-CAM map at `denseblock4.denselayer16.conv2` for each image's ground-truth class; the map is ReLU'd, resized to 224x224 and max-normalised. A separate U-Net (`MultiTaskUNet`, four `DoubleConv` blocks, 64 to 512 channels, 7,705,224 parameters) is trained on those maps as soft targets with Dice+BCE loss (Adam 3e-4, batch 8, 3,000 images, 20 epochs).
- It states what the narrow channel costs: the U-Net mask agrees with a fresh Grad-CAM map at a median Dice of 0.22, with 27% of films showing no overlap (new histogram figure). The 0.48 Dice in the report was a soft-mask score on the segmentation validation split and is no longer the headline.
- It states what the compression buys: one forward pass instead of forward plus backward at serving time (see Comment 5).
- The Fig. 3 caption and diagram mark the stop-gradient and say that the only channel is one 224x224 scalar field per image.

### Comment 5: How long does the CPU-only web app take to return the heatmap?

**Problem in the report.** The report only said "under a second".

**Fixes made.** Section 5 and Appendix E now report measured CPU latency (Intel i7-13620H, 10 threads, single request, 40-request run):

| Path | Share of test films | Mean (sd) |
| --- | --- | --- |
| Distilled U-Net (one forward pass) | 91% | 293 ms (81) |
| Live Grad-CAM fallback (forward + backward) | 9% | 560 ms (22) |
| Weighted by the test-set mix | 100% | 318 ms |

Report generation is a dictionary lookup and adds no measurable time. The roughly 2x gap between the two paths is the practical reason the U-Net exists.

**Still to do.**
- [ ] Once a VLM or LLM generator (S2 to S4) is added, report its latency separately, including network time for any hosted API.

---

## Other changes made during the revision

- Moved to the official NeurIPS 2026 style file with `[dblblindworkshop]` and `\workshoptitle{...}`; authors and acknowledgements are anonymised.
- Body is 7 pages (limit 9); references, checklist and appendices are extra.
- Every repository-derived number was fact-checked against `chest_classifier_metrics.json` and the analysis scripts. One correction: the 98.5% recall belongs to Normal, not tuberculosis.
- Added a Broader Impact section, and a limitation stating that the summary is not a clinical finding and the interface should say so.

## Pre-submission checklist

- [ ] Run at least one VLM cell (S3 or S4) and S2 (Comment 3)
- [ ] H1 vs H3 per-film split (Comment 2)
- [ ] `paired_summary.pdf` and restore or replace the interface figure (Comments 1 and 2)
- [ ] State the S1 x H0 number in the text (Comment 2)
- [ ] Replace the abstract with the version below, then update its results sentence if VLM numbers are added
- [ ] Anonymise the code link (for example anonymous.4open.science) for review; use the real GitHub URL only in the camera-ready
- [ ] Confirm the double-blind workshop track on the OpenReview submission page
- [ ] Recompile in Overleaf and check page count

---

## Amended abstract

The abstract follows the pattern of the reference abstract Dr Imad shared: it opens with the broad field and its clinical motivation, narrows to the specific problem and why it matters clinically, states what "this work" proposes and on what data, gives the key quantitative results, closes on practical scope, and ends with a code availability line. It is one paragraph with abbreviations defined at first use.

> Deep learning (DL) has transformed the interpretation of chest radiographs (CXRs) over the past decade, and automated systems for pneumonia, tuberculosis (TB) and COVID-19 are now deployed as triage tools in settings with too few radiologists. Convolutional neural networks (CNNs) paired with Gradient-weighted Class Activation Mapping (Grad-CAM) are the usual way of showing a clinician where the evidence for a prediction lies, and deployed systems increasingly place a short natural-language summary beside the prediction and its heatmap. A reader naturally takes the heatmap as proof that the summary is grounded in the image, yet whether the summary actually depends on the localisation is rarely tested, and the text-overlap metrics used to score generated reports reward fluency rather than image support. For a generated report to serve as a reasoning aid, its spatial claims must change when the evidence changes and must be correct when they are made. In this work we audit a deployed, CPU-only four-class CXR system (DenseNet-121 classifier; Grad-CAM compressed into a U-Net for single-pass serving), trained on eight public sources and reaching 96.18% accuracy on a patient-grouped held-out set of 4,759 images. Its summary turns out to be a static template keyed only on the predicted class, so no property of the heatmap reaches the text. We then condition the summary on measured region descriptors (side, zone, extent and focality) behind an in-lung gate, and compare summaries with and without localisation on 245 scored test radiographs. The gate withholds a location claim on 62% of films because the served heatmap lies outside the lung fields, and giving each film another film's heatmap leaves the claim rate unchanged while grounding accuracy falls to 15%. The full pipeline returns a prediction, heatmap and summary in a mean of 318 ms on a CPU-only laptop. We propose an evaluation protocol that crosses heatmap source with generator type, including vision language models (VLMs), as a practical check on whether a generated radiology report reads the image it describes. Code is available at: [anonymised link for review].

*About 330 words. If the VLM cells are run before submission, replace the second-last sentence's "We propose an evaluation protocol..." with the measured VLM result (for example "A VLM given the overlay makes spatial claims on X% of films, of which Y% are grounded, against Z% under the permuted control"), and cut the classifier clause if space is needed.*
