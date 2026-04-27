# pediatric-leukemia-gan
Privacy-preserving synthetic genomic AI for pediatric leukemia research
 *The Problem
To build AI that detects pediatric leukemia early, researchers need thousands of children's genomic records.
But real pediatric genomic data is:

 Locked behind 12+ months of ethics approvals
 Protected by HIPAA, GDPR, and hospital privacy laws
 Permanently sensitive — DNA cannot be changed like a password
 Dangerous if exposed — reveals entire family's identity forever

A 2019 NIH study confirmed that accessing genomic data takes months of approvals — and pediatric cancer data is the hardest to access of all.
Result: Researchers wait. AI never gets built. Children get detected late. Survival rates stay low.

 The Solution — SnakeTail
SnakeTail generates unlimited synthetic pediatric leukemia patients using a custom Generative Adversarial Network (GAN) — without touching a single real child's DNA.
How it works:
Step 1: Create 500 fictional patients
        using real mutation frequencies
        from Nature Genetics 2024 (887 pAML patients)
        
Step 2: Train custom GAN on those 500 patients
        Generator learns leukemia mutation patterns
        Discriminator learns to spot fakes
        They fight for 1000 epochs
        
Step 3: Generate 1000 new synthetic patients
        Apply post-hoc frequency correction
        Match real clinical rates exactly
        
Step 4: Train RandomForest on synthetic data
        94.5% accuracy (V1) → 98.5% accuracy (V2)
        
Step 5: Doctors input symptoms or genomic data
        Get instant leukemia risk prediction
        Zero real patient DNA ever used

 The 8 Leukemia Mutations Tracked
Frequencies sourced from Nature Genetics (2024) — 887 pediatric AML patients
MutationFrequencyWhat it doesFLT3-ITD35% of AMLCell growth OFF switch brokenBCR-ABL25% of ALLPhiladelphia chromosome fusionTP5320% high riskTumour suppressor brokenRUNX125% of AMLBlood cell development blockedDNMT3A22% of AMLGene regulation disruptedNPM130% of AMLDNA repair compromisedCEBPA15% of AMLWhite cell maturation blockedWT118% poor prognosisAggressive disease marker

 File Structure
pediatric-leukemia-gan/
│
├── apptest.py              # Main Streamlit web application (5 tabs)
├── gan.py                  # Custom GAN engine (Generator + Discriminator)
│
├── genome.py               # V1 dataset generator (500 patients, 8 mutations)
├── genomic_gan.py          # V1 GAN training pipeline + frequency correction
├── risk_predictor.py       # V1 RandomForest predictor (94.5% accuracy)
│
├── genome_v2.py            # V2 dataset generator (500 patients, 28 features)
├── genome_gan_v2.py        # V2 GAN training pipeline
├── risk_predictor_v2.py    # V2 RandomForest predictor (98.5% accuracy)
│
├── genomic_dataset.csv     # V1 original 500 patients
├── synthetic_genomes.csv   # V1 GAN output 1000 patients
├── genomic_v2_dataset.csv  # V2 original 500 patients
├── synthetic_genomes_v2.csv# V2 GAN output 1000 patients
│
├── leukemia_model.pkl      # V1 trained model
├── leukemia_model_v2.pkl   # V2 trained model
│
├── images/                 # Screenshots and diagrams
└── README.md               # This file

 GAN Architecture
Generator (85,004 parameters)
noise(128) → Linear(128→256) + BatchNorm + ReLU
           → Linear(256→128) + BatchNorm + ReLU
           → Linear(128→output_dim) + Tanh
Discriminator (67,713 parameters)
input → Linear(input→128) + LeakyReLU(0.2) + Dropout(0.3)
      → Linear(128→256)   + LeakyReLU(0.2) + Dropout(0.3)
      → Linear(256→128)   + LeakyReLU(0.2)
      → Linear(128→1)     + Sigmoid
Training

Optimizer: Adam (lr=0.0002, betas=(0.5, 0.999))
Loss: Binary Cross Entropy
Epochs: 1000
Batch size: 32


 Results
VersionFeaturesAccuracyKey AdditionV18 binary mutations94.5%Core GAN pipelineV228 multi-modal features98.5%Gene expression + chromosomal + clinical markers
Key clinical differences learned by GAN:
WBC count:       Leukemia 84,221 vs Healthy 7,576  (11x difference)
Blast %:         Leukemia 64.7%  vs Healthy 1.5%
FLT3 expression: Leukemia 8.5    vs Healthy 3.2    (39x more active)
Mutation burden: Leukemia 18.3   vs Healthy 1.5

 Web Application — 5 Tabs
Tab 1 — Generate Synthetic DNA
Generate unlimited synthetic pediatric leukemia patients for research
Tab 2 — Symptom Journey
Doctor inputs symptoms (no lab results needed)
→ Biological explanation
→ Probable mutations with probability %
→ Recommended tests to order
→ Next clinical steps
Tab 3 — Predict V1 (94.5%)
Upload patient CSV file → instant prediction
OR manual checkbox input
→ Visual mutation bars (red = present)
→ Clinical summary with progress bars
→ Risk % with colour-coded banner
→ Plain English clinical interpretation
Tab 4 — Predict V2 (98.5%)
28 feature inputs:
→ Gene expression sliders (log2 TPM scale)
→ Chromosomal abnormality checkboxes
→ Blood test values (WBC, blast %, LDH)
→ Treatment response sliders
→ Auto-warnings for critical values
Tab 5 — Hidden Patterns
Statistical knowledge extracted from GAN:
→ Mutation co-occurrence rates
→ Age risk bands
→ Mutation burden thresholds
→ Dangerous mutation combinations
→ Downloadable research report

This is what the application look like
## App Homepage
![Homepage]<img width="1913" height="1077" alt="Snaketail" src="https://github.com/user-attachments/assets/2287d31c-c604-4d8c-9c7f-052924738c7c" />

### Symtoms
![Symtoms]<img width="1914" height="1050" alt="Symtoms" src="https://github.com/user-attachments/assets/cd34a4a8-0669-4743-b6eb-04a85fd96c6e" />

### Mutation Status Bars
![Mutations]<img width="1904" height="1071" alt="mutations (2)" src="https://github.com/user-attachments/assets/a8398017-5527-4517-aa36-d5bf24aaf34d" />

### Detect Risk
![Symptoms]<img width="1905" height="1076" alt="Detect Risk" src="https://github.com/user-attachments/assets/76a9f10b-0497-458f-81bf-72caa9642779" />

### Hidden Patterns
![Insights]<img width="1838" height="1061" alt="insight1" src="https://github.com/user-attachments/assets/60ee2096-18c7-4f36-ada8-7750e435a8e4" />
<img width="1916" height="1074" alt="Insight3" src="https://github.com/user-attachments/assets/99f7f4a9-d438-446d-a2d1-2ca0b7e1760a" />
<img width="1915" height="988" alt="insight2" src="https://github.com/user-attachments/assets/b8d72a13-6569-43bf-abf0-3d83734a581e" />

How GAN works?
### GAN Pipeline
![GAN Pipeline]<img width="1163" height="939" alt="Diagram2" src="https://github.com/user-attachments/assets/688d4cc4-97e2-4084-895c-0f33056f5870" />

### Clinical Flow
![Clinical Flow]<img width="1395" height="936" alt="Diagram1" src="https://github.com/user-attachments/assets/af093b3d-5669-4cb4-b6b1-6da7dad01a5a" />


Installation
Prerequisites
bashPython 3.10+
pip
Install dependencies
bashpip install streamlit pandas numpy torch scikit-learn biopython
Run the application

Key Innovations
1. Post-hoc Frequency Correction
GAN trained on inflated frequencies for stability → corrected back to real clinical rates after generation
python# Inflate for GAN training stability: FLT3 = 55%
# Correct after generation to real rate: FLT3 = 35%
2. Symptom-to-Genomic Journey
First tool to guide doctors from observable symptoms → probable mutations → genomic confirmation → treatment plan
3. Hidden Pattern Extraction
GAN-learned statistical patterns shared as medical insights — publishable, citable, privacy-safe
4. Multi-modal V2 Features
5 feature layers: mutations + gene expression (RNA-seq derived) + chromosomal + clinical markers + treatment response

 References
Nature Genetics (2024) — "A new genomic framework to categorize pediatric AML" — 887 pAML patients
https://www.nature.com/articles/s41588-023-01640-3
NIH/PMC (2019) — "Barriers to accessing public cancer genomic data"
https://pmc.ncbi.nlm.nih.gov/articles/PMC6586850
Pediatric Research/Nature (2025) — "Breaking barriers: fostering equitable access to pediatric genomics"
https://www.nature.com/articles/s41390-025-03859-8
WHO (November 2024) — "New principles for ethical human genomic data sharing"
https://www.who.int/news/item/20-11-2024
Leukemia Journal (2025) — "Real-time genomic characterization of pediatric acute leukemia"
https://www.nature.com/articles/s41375-025-02565-y




