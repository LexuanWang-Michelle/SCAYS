# SCAYS — Strong-Context Annotation for Youth Sentiment

> **A Multi-Dimensional Sentiment Corpus for Chinese Adolescent Social Media**
>
> Building data from identity, temporality, relationships, and subculture, not just keywords.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Language: Chinese](https://img.shields.io/badge/Language-Chinese-red.svg)](#)
[![Data: Xiaohongshu](https://img.shields.io/badge/Source-Xiaohongshu-ff2442.svg)](#)
[![Model: BERT](https://img.shields.io/badge/Model-BERT--base--chinese-blue.svg)](#)

---

[中文版](README.md) | [English Version](README_EN.md)

## TL;DR

**SCAYS 1.0** is a Chinese adolescent sentiment corpus collected from Xiaohongshu (RED), currently containing **~4,000** sentence-level annotated data samples.

Existing Chinese sentiment datasets are almost entirely "flat"—they scrape random texts, label them as positive/negative/neutral, and call it a day. They don't care *who* is speaking, *what stage* they are in, or *what relationships* they are facing. However, adolescent emotions never exist in isolation; they grow within specific **Identities, Temporalities, Relationships, Cultures, Bodies, Economics, and Behaviors**.

The biggest difference between SCAYS and other datasets: **We designed our collection strategy based on these seven dimensions from the very beginning**. We didn't collect data first and apply tags later; we established the dimensional framework first, and then conducted targeted collection around the keywords of each dimension. This shift from **"flat data" to "multi-dimensional contextualization"** is the core reason for SCAYS's existence.

---

## Why Do We Need SCAYS?

### The Fundamental Flaw of Traditional Datasets: Flatness from the Start

The problem with traditional sentiment datasets isn't just the annotation method; **it's the loss of context right from the data collection phase**:

| | Traditional Datasets | SCAYS |
|------|---------|-------------|
| **Collection Method** | General keyword scraping or random sampling | **Targeted collection via a 7D framework**: Each dimension has specifically designed keywords (100+ in total). |
| **Data Perspective** | A sentence is just a sentence, lacking context | A sentence + Who is speaking (Identity) + When (Temporality) + In what context (Relational Field) |
| **Annotation Depth** | Positive / Negative / Neutral | Emotion Type + Emotion Motivation + Situated Context Dimension |
| **Language Coverage** | General Chinese | Exclusively covers Chinese middle/high school students' discourse and subcultural expressions. |

In short: **Traditional datasets optimize the annotation phase; SCAYS layers the data directly from the source.**

### Implicit Emotion Recognition: A Means, Not an End

Through this process, SCAYS also addresses a typical shortcoming of traditional models—implicit emotions. The authentic expressions of teenagers often contain no explicit emotion words:

```text
"My mom forces me to eat an apple every day. I never eat it, but she still gives it to me every single time."
→ No emotion words, but reveals complex feelings within family relationships.

"My grades are actually okay, but with this environment, I can probably foresee my future scores."
→ No emotion words, but reveals anxiety and self-doubt about the future.
```

Traditional models would classify all of these as "neutral". But identifying implicit emotions isn't the primary goal of SCAYS—it is simply a natural byproduct of multi-dimensional contextual annotation. **What truly matters is that because we collect data dimensionally, these types of sentences are not overlooked during the collection phase.**

---

## The 7-Dimension Contextual Annotation Framework

SCAYS believes that adolescent emotions are not isolated "positive/negative" labels, but are generated within specific contexts of **Identity, Temporality, Relationships, Culture, Bodies, Economics, and Behaviors**. These seven dimensions are emotionally neutral by default—each can produce positive, negative, or complex intertwined feelings. We designed them not just to catch "bad moods", but to restore the complete living context of teenagers:

### Dimensions Overview

| # | Dimension | Core Focus |
|---|-----------|------------|
| 1 | **Identity Anchor** | Who are you? — Grade, school type, academic track, campus status |
| 2 | **Temporal Rhythm** | When is it? — Exams, daily routines, vacations, study cycles |
| 3 | **Relational Field** | With whom? — Family, teachers, peers, romance |
| 4 | **Digital Refuge** | Escaping to where? — Fandoms, anime/merch circles (Guquan), gaming, OC (Original Character) circles |
| 5 | **Somatic Awakening** | What's happening to the body? — Appearance, physiological changes, sleep, illnesses |
| 6 | **Economic Imprint** | Behind the spending? — Family conditions, financial control, material comparisons |
| 7 | **Agency Reclaim** | How to rebel? — Makeup, skipping school, truancy, and other rebellious behaviors |

### Detailed Tags by Dimension

> ⚠️ The following only shows **representative tag examples** for each dimension. The full keyword list contains 100+ tags and will be released with dataset v2.0.

<details>
<summary>Dimension 1: Identity Anchor — 4 subcategories, 17 tags</summary>

Adolescent emotional expression highly depends on their identity. "I can't finish studying" carries a vastly different anxiety level for a middle schooler versus a high school repeater.

| Subcategory | Example Tags |
|------|---------|
| Grade/Stage | `Pre-Senior` (准高三), `Repeater` (复读生), etc. (5 tags) |
| Routine | `Boarding student` (住校生), etc. (3 tags) |
| Track | `STEM student` (理科生), `Art student` (美术生), etc. (4 tags) |
| Status | `Top student` (学霸), `Underachiever` (学渣), etc. (5 tags) |
</details>

<details>
<summary>Dimension 2: Temporal Rhythm — 4 subcategories, 24 tags</summary>

Teenage emotions are highly cyclical: anxiety spikes before monthly exams, erupts at the end of holidays, and peaks on report card day.

| Subcategory | Example Tags |
|------|---------|
| Exams | `Monthly Exam` (月考), `Report Card` (发成绩单), etc. (8 tags) |
| Daily | `Evening Self-study` (晚自习), `Teacher keeping class late` (拖堂), etc. (6 tags) |
| Vacations | `Post-holiday syndrome` (开学综合征), etc. (5 tags) |
| Study | `Unbalanced grades` (偏科), `Small-town swot` (小镇做题家), etc. (6 tags) |
</details>

<details>
<summary>Dimension 3: Relational Field — 4 subcategories, 30 tags</summary>

Teenagers cannot choose their family, teachers, or classmates. Relationships involve both conflict (isolation, favoritism) and growth. This "unchangeable" nature makes emotional experiences exceptionally intense.

| Subcategory | Example Tags |
|------|---------|
| Family | `Favoritism` (偏心), `Checking phones` (查手机), `Moral kidnapping` (道德绑架), etc. (9 tags) |
| Teachers | `Targeted` (被针对), `Called parents` (请家长), `School-wide criticism` (全校通报), etc. (10 tags) |
| Peers | `Isolated` (孤立), `School cold violence` (校园冷暴力), etc. (8 tags) |
| Romance | `Secret crush` (暗恋), `Caught dating` (早恋被抓), etc. (3 tags) |
</details>

<details>
<summary>Dimension 4: Digital Refuge — 4 subcategories, 21 tags</summary>

Teenagers build their spiritual belonging in the digital world. The sense of belonging from chasing stars or gaming achievements are real positive emotions, but these spaces also create negative fluctuations (scandals, quitting fandoms, being scammed). Traditional NLP models rarely understand this subcultural jargon.

| Subcategory | Example Tags |
|------|---------|
| Fandom | `Scandal/House collapse` (塌房), `Unstanning` (脱粉), etc. (7 tags) |
| Anime/Merch | `Buying merch` (吃谷), `Quitting the pit` (退坑), etc. (5 tags) |
| Gaming | `Anti-addiction system` (防沉迷), `Pity pull` (抽卡保底), etc. (6 tags) |
| OC/Design | `OC` (Original Character), etc. (3 tags) |
</details>

<details>
<summary>Dimension 5: Somatic Awakening — 4 subcategories, 16 tags</summary>

Puberty brings drastic physical changes, causing both distress (acne, insomnia) and pleasant surprises (growing taller). This dimension focuses on the complex feelings toward physical changes.

| Subcategory | Example Tags |
|------|---------|
| Appearance | `Acne` (青春痘/烂脸), `Height` (身高), etc. (4 tags) |
| Physiology | `Menstrual cramps` (痛经), `Voice changing` (变声期), etc. (4 tags) |
| Other | `Sleep` (睡眠), `Eyesight` (视力), etc. (3 tags) |
| Illness | `Somatization` (躯体化), `Eating disorders` (进食障碍), etc. (5 tags) |
</details>

<details>
<summary>Dimension 6: Economic Imprint — 3 subcategories, 12 tags</summary>

Consumption behaviors reflect family background and peer culture, encompassing the blow to self-esteem from material gaps, as well as the satisfaction of buying desired merch.

| Subcategory | Example Tags |
|------|---------|
| Family Status | `Study abroad` (留学), etc. (3 tags) |
| Economic Control | `Allowance` (零花钱), `Broke from buying merch` (买谷吃土), etc. (4 tags) |
| Material Comparison | `Fake shoes` (假鞋), etc. (5 tags) |
</details>

<details>
<summary>Dimension 7: Agency Reclaim — 6 tags</summary>

Teenagers reclaim autonomy under high-pressure environments through "deviant behaviors," which serve as emotional outlets and important risk signals.

| Example Tags |
|---------|
| `Skipping school` (逃学), `Cutting class` (翘课), etc. (6 tags) |
</details>

### The 7 Dimensions Are Coordinates, Not Paths

These seven dimensions are not a causal chain, but **seven cross-sections describing the world a teenager lives in**:

```
        Identity  — Who are you? (Grade, routine, track, campus status)
        Temporal  — When is it? (Exams, daily routine, vacation, study state)
        Relation  — With whom? (Family, teachers, peers, romance)
        Refuge    — Where is your spiritual world? (Fandom, merch, gaming, OC)
        Somatic   — What is your body experiencing? (Appearance, physiology, sleep, illness)
        Economic  — What are your material conditions? (Family status, control, comparison)
        Reclaim   — What did you do? (Deviant behaviors & reclaiming autonomy)
```

At any given moment, a teenager exists simultaneously at some position on these seven axes. The combination of these positions is their "Context."

For instance, the same phrase "I don't want to live anymore" means entirely different things in different coordinate combinations:
- **Identity** = Senior + **Temporal** = Report card day → Likely post-exam emotional venting.
- **Identity** = Repeater + **Relation** = Isolated + **Somatic** = Chronic insomnia → A serious cry for help requiring intervention.

### The 7 Dimensions Are Basis for Deduction

On social media, users rarely state their full identity—but they might say things only specific identities would experience. Here, the seven dimensions can **mutually deduce and fill in missing information**:

```text
Example: A post mentioning a specific daily routine

Known Dimensions:
  Temporal = A specific daily school routine
  Temporal = A specific teacher-student interaction
  
Deducible Dimensions:
  Identity → Specific school type (implied by routine)
  Identity → Specific academic track (implied by subject keywords)
  Identity → Study state (tone implies long-term difficulty)
  Relation → Teacher-student pressure (passive endurance)
```

This means the 7D framework solves not only "data flatness" but also the **"identity ambiguity" on social media**.

---

## Current Data Overview (v1.0)

### Data Scale

| Category | Count | label | Description |
|------|------|-------|------|
| Explicit Emotion (normal) | ~1,800 | 1 | Contains explicit emotion words |
| Implicit Emotion (invisible) | ~700 | 1 | No emotion words but implies emotion |
| Neutral (neutral) | ~1,500 | 0 | No emotional expression |
| **Total** | **~4,000** | | |

### Data Source

- **Platform**: Xiaohongshu (RED)
- **Scope**: Targeted keywords designed based on Dimension 1 (Identity Anchor)
- **Type**: Post titles, bodies, and comments (split at sentence-level)
- **Method**: DrissionPage simulating human browsing to preserve authentic linguistic styles.

### Example Data

```csv
Keyword,Source,Sentence,label,Implicit Emotion,Emotion Motivation
准初三,content,快中考了这种成绩咋办🥹🥹,1,Anxiety,Academics
住宿生,comment,我去我感冒一个多星期了导员不给请假一直硬抗着,1,Grievance,Body
复读生,comment,复读生和应届生同分同录无隐形限制,0,Neutral,None
学渣,comment,我成绩其实还行但再这个环境我大概能看到自己以后的成绩了,1,Anxiety,Academics
```

---

## Roadmap

### Near-term (v1.x)
- [ ] Expand data scale: from ~4,000 to 20,000+ samples.
- [ ] Expand collection keywords: Cover dimensions 2-7.
- [ ] Improve implicit emotion annotation: LLM assistance + human double-checking.
- [ ] Release pre-trained BERT classifier weights.

### Mid-term (v2.0)
- [ ] **Full 7D Annotation**: Tag each sentence with its specific dimension and sub-tags, not just emotion types.
- [ ] **Multi-platform Expansion**: Expand from Xiaohongshu to Bilibili, NetEase Cloud Music, etc.
- [ ] **Dynamic Lexicon**: Continuous keyword updates and expansion.
- [ ] **Benchmarks**: Systematically compare SCAYS with mainstream Chinese sentiment datasets.

### Long-term (v3.0)
- [ ] **Multimodal Fusion**: Integrate memes, short video soundtracks, etc.
- [ ] **Emotion Tracing**: Don't just detect *if* there's an emotion, infer *where it comes from* via 7D causal chains.
- [ ] **Domain-Specific Fine-Tuned Models**: Release dedicated models for Chinese adolescent social pressures.

---

## Technical Philosophy

Existing Chinese sentiment datasets suffer from four structural flaws. SCAYS provides clear countermeasures for each:

### ❶ Scarcity — Adolescent sentiment datasets are nearly non-existent.
> **SCAYS's Countermeasure**: Build a sentiment corpus from scratch specifically for Chinese middle and high school students, using keywords drawn from their authentic discourse and life scenarios.

### ❷ Flat Data
> **SCAYS's Countermeasure**: Establish a 7D contextual framework first → collect based on dimensional keywords → annotate emotion type + motivation + dimension. Data is structurally contextualized from the source.

### ❸ Identity Ambiguity
> **SCAYS's Countermeasure**: The 7D framework allows dimensions to deduce and fill in missing info. Even if users don't state their identity, missing contexts can be deduced from known dimensions.

### ❹ Semantic Drift
Youth language evolves faster than any other group. Subcultural slang constantly diffuses into the mainstream, shifting semantics.
> **SCAYS's Countermeasure**: Track slang evolution and establish a dynamic lexicon updating mechanism. The dataset is a living document tracking youth discourse, not a one-time delivery.

---

## Quick Start

### Requirements
```bash
Python >= 3.8
torch >= 1.13
transformers >= 4.20
pandas
scikit-learn
```

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/SCAYS.git
cd SCAYS
pip install -r requirements.txt
```

### Loading Data
```python
import pandas as pd

# Load dataset
df = pd.read_csv("data/训练数据集.csv")

# View distribution
print(df["label"].value_counts())
# 1 (Emotional): ~2500
# 0 (Neutral): ~1500
```

### Training the BERT Classifier (3-Class)
```bash
python src/bert_3class.py --train
```

### Predicting New Data
```bash
python src/bert_3class.py --predict --input to_predict.csv
```

---

## Project Structure

```
SCAYS/
├── data/
│   ├── normal.csv              # Explicit emotion sentences
│   ├── invisible_labeled.csv   # Implicit emotion annotations
│   ├── 训练数据集.csv            # Merged full training set
│   └── bert训练集.csv           # Ready-for-BERT format (raw_text, label)
├── src/
│   ├── bert_3class.py          # BERT training/predicting script
│   ├── filter_emotion.py       # Data cleaning & emotion routing
│   ├── label_invisible.py      # Implicit emotion labeling
│   └── merge_review.py         # Human review merging script
├── README.md
├── README_EN.md
├── requirements.txt
└── LICENSE
```

---

## Ethics Statement
- All data is sourced from publicly accessible social media content.
- **Username de-identification** has been applied; no personal identifiable information is included.
- This dataset is strictly for **academic research** purposes. Commercial surveillance or user profiling is strictly prohibited.
- If identifiable personal information is found, please contact us for removal.

---

## License
This dataset is released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
- ✅ Academic Research
- ✅ Annotation Methodology Reference
- ❌ Commercial Use
- ❌ User Surveillance/Profiling

---

## Contributing
SCAYS is a **living foundation**. We need:
- **Data Miners**: To uncover raw emotional expressions deep in network corners.
- **Annotation Experts**: To define emerging subcultural tags with AI synergy.
- **Developers**: To achieve SOTA performance on Chinese adolescent sentiment analysis.

---

## Acknowledgements
- Data Source: [Xiaohongshu (RED)](https://www.xiaohongshu.com)
- Base Model: [BERT-base-Chinese](https://huggingface.co/bert-base-chinese)
- Collection Tool: [DrissionPage](https://github.com/g1879/DrissionPage)