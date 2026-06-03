# LLM-Based Medical Science Content Classification

A pipeline for classifying Chinese medical/health science articles from WeChat public accounts using LLM (DeepSeek) and BERTopic.

## Pipeline Overview

```
step1_identification.py  → Step 1: Identify health-related content (is/isn't health science)
step2_bertopic.py        → Step 2: BERTopic exploratory clustering (title + first 500 chars)
step3_topic_classification.py → Step 3: 15-category topic taxonomy classification
step4_framing_classification.py → Step 4: 5-category framing type classification
```

## Requirements

```
pip install pandas openpyxl tqdm requests
pip install bertopic sentence-transformers umap-learn hdbscan
pip install openai  # for DeepSeek API via SiliconFlow
```

## Environment Variables

```bash
export DEEPSEEK_API_KEY="your-api-key"
export DEEPSEEK_MODEL="deepseek-ai/DeepSeek-V4-Flash"  # optional
```

## Step Descriptions

### Step 1: Identification
Classifies whether an article is health-related science content based on its title.

### Step 2: BERTopic Clustering
Exploratory semantic clustering on a random sample (default 5000), using title + first 500 characters.

### Step 3: Topic Classification
15-category taxonomy using DeepSeek LLM:
1. Cancer and Tumor Prevention & Treatment
2. Chronic Disease and Metabolic Disease Management
3. Cardiocerebrovascular Diseases and Emergency Warning
4. Respiratory and ENT Health
5. Infectious Disease Prevention and Control
6. Maternal, Pediatric and Adolescent Health
7. Organ Transplantation and Special Medical Procedures
8. Traditional Chinese Medicine and Wellness
9. Nutrition and Lifestyle
10. Musculoskeletal and Rehabilitation
11. Mental and Sleep Health
12. Environmental and Preventive Health
13. Reproductive and Sexual Health
14. Oral Health
15. Other

### Step 4: Framing Classification
5-category framing analysis:
1. Informational framing
2. Behavioral guidance framing
3. Warning/risk framing
4. Narrative framing
5. Emotional reassurance framing
