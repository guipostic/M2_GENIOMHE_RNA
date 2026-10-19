# RNA Bioinformatics Research Projects — M2 GENIOMHE-AI

Project briefs and an introductory primer for the research projects of the Master 2 **GENIOMHE-AI** course in RNA bioinformatics at **Université Paris-Saclay**, academic year 2026–2027.

Author: Guillaume Postic

## Contents

| File | Description |
| --- | --- |
| [`slides/RNA_Bioinformatics_Primer_for_Project_Choice.pdf`](slides/RNA_Bioinformatics_Primer_for_Project_Choice.pdf) | Primer slides (33 slides): the background needed to choose a project |
| [`briefs/RNA_Bioinformatics_M2_Project_Briefs.pdf`](briefs/RNA_Bioinformatics_M2_Project_Briefs.pdf) | The 16 project briefs (48 pages, A4) |
| [`briefs/RNA_Bioinformatics_M2_Project_Briefs.docx`](briefs/RNA_Bioinformatics_M2_Project_Briefs.docx) | Editable version of the briefs |

Start with the slides, then read the briefs.

## Scientific ambition

These are research projects, not training exercises. Each one aims at a scientifically rigorous and potentially novel result that, if the work succeeds, could contribute to the literature and form the basis of a manuscript submitted to a peer-reviewed journal.

Publication is not guaranteed: it depends on the novelty, robustness and significance of the results, and on the quality of the execution. Nor is it a requirement for passing the course. A well-conducted project that yields a meaningful negative result or a useful research resource is a successful project.

## The projects at a glance

Difficulty is rated for a part-time team over one semester: **2** means mature tools and a clear evaluation, **3** moderate, **4** several hard components and a scope that must be cut early. Team sizes are recommendations, not limits.

| # | Project | Difficulty | Compute | Recommended team |
| ---: | --- | :---: | --- | :---: |
| 1 | Temporal, homology-stratified test of 2026 co-folding models | 4 | GPU, heavy | 4–5 |
| 2 | Docking vs co-folding for RNA–small-molecule complexes | 4 | GPU, moderate | 4–5 |
| 3 | Riboswitch conformational switching in co-folding models | 3 | GPU, light | 3–4 |
| 4 | Modified nucleotides as input to structure predictors | 2 | GPU, light | 2–3 |
| 5 | Stereochemistry and refinement of deep-learning RNA models | 2 | CPU | 2–3 |
| 6 | Model selection in the co-folding era | 3 | CPU (GPU optional) | 3 |
| 7 | Calibration of base-pair probabilities, and their relation to chemical probing | 2 | CPU | 2–3 |
| 9 | MSA ablation for RNA co-folding | 3 | CPU heavy, then GPU | 3–4 |
| 10 | Leakage audit of RNA language-model benchmarks | 2 | CPU (GPU optional) | 3–4 |
| 11 | Replication study of a recent RNA machine-learning paper | 3 | Depends on the paper | 2–4 |
| 12 | Systematic review of evaluation practice in RNA machine learning | 2 | None | 4–6 |
| 14 | A harmonized, confidence-tiered evidence atlas of RNA modification sites | 3 | None | 3–4 |
| 15 | Rfam's structural dark matter | 3 | GPU, moderate | 3–4 |
| 16 | Independent evaluation of 3D RNA inverse-folding designs | 4 | GPU, heavy | 4–5 |
| 17 | Probing RNA language-model embeddings for structural annotations | 3 | GPU, light | 3 |
| 18 | Viral RNA elements with 2026 structure tools | 3 | GPU, light | 3–4 |

Numbering follows the original project list. Projects 8 and 13 were withdrawn because neither had a scaled-down version that would stand on its own as a result.

## What each brief contains

- The research question and its hypothesis
- Data, tools and evaluation metrics
- A 16-week work plan
- A risk table, with a fallback for each risk
- A scope table: the **required** work (the minimum viable version, due in week 8), one **main extension**, and **stretch** goals that are never an implicit requirement
- A novelty-and-reproducibility block: closest prior work, specific contribution, main endpoint, evaluation risks and feasibility evidence

## What the primer covers

The 33 slides fall into six parts:

1. **RNA and its structure**: RNA classes, levels of structure, dot-bracket notation, tertiary motifs, the nearest-neighbour energy model, conformational ensembles.
2. **Measurement and data**: experimental structures and chemical probing, sequencing technologies, databases, file formats.
3. **Prediction**: homology search, covariation, secondary- and tertiary-structure prediction, and what blind assessments say about the state of the art.
4. **Judging predictions**: structural metrics, benchmark design, machine-learning conventions, statistical reporting, RNA language models.
5. **Specialised topics**: RNA modifications, RNA–small-molecule interactions, RNA design.
6. **Practice and choosing**: the toolbox, how the semester runs, a map from topics to briefs, a glossary and a reading list.

## How the semester runs

| Week | Milestone |
| :---: | --- |
| 1 | Roles assigned, environment installed, and an end-to-end smoke test on three representative inputs, recording runtime, memory and storage for GPU projects |
| 2 | Literature refresh for the exact question; it seeds the report's introduction |
| 3 | Protocol frozen: tools, versions, data releases, metrics, hypotheses and statistical tests. It becomes the first draft of the methods section |
| 8 | Minimum viable version reviewed; a team that is behind cuts scope here |
| 14 | Results frozen |
| 15–16 | Final report, repository and presentation |

## Deliverables

Every project delivers:

- a versioned code repository with environment files, run scripts and result tables;
- the protocol frozen in week 3, with its change log;
- the scope table and the completed novelty-and-reproducibility block;
- a final report written as a research article (abstract, introduction, methods, results, discussion and references), with figure legends, a data and code availability statement, and a supplementary table of every tool version and dataset release used;
- a 15-minute presentation.

## Choosing a project

1. Read the primer slides, then the 16 project briefs.
2. Form a team and rank three briefs, with one sentence on why each fits you.
3. Send your ranking by the end of week 1.

Questions about GPU access: guillaume.postic@universite-paris-saclay.fr

## A note on currency

RNA structure prediction tools change quickly. Statements in the briefs about tool inputs (for example, which predictors accept RNA multiple sequence alignments) and about model training cutoffs were checked on 16–17 October 2026 against the documentation cited in the briefs' reference section. Teams re-check them in week 1 against the exact releases they install.

## How to cite

Postic G. (2026). *RNA bioinformatics research projects: briefs and primer.* Master 2 GENIOMHE-AI, Université Paris-Saclay.

## Licence

[Licence to be chosen. For teaching material, Creative Commons Attribution 4.0 International (CC BY 4.0) is a common choice.]

The Université Paris-Saclay logo and graphic charter used in the slides belong to the university and are not covered by this licence.
