import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


# VERSION 2 — MULTI-MODAL GENOMIC DATASET
# Adds gene expression + chromosomal data


def generate_v2_dataset(n_patients=500):
    print("Generating Version 2 multi-modal genomic dataset...")
    np.random.seed(42)

    n_leukemia = int(n_patients * 0.5)
    n_healthy  = n_patients - n_leukemia

    # LEUKEMIA PATIENTS
    leukemia = pd.DataFrame({

        # Demographics
        'age':    np.random.randint(1, 18, n_leukemia),
        'gender': np.random.choice(['M', 'F'], n_leukemia),

        # Layer 1 — Binary mutations (V1)
        'FLT3_ITD':  np.random.choice([0,1], n_leukemia, p=[0.45, 0.55]),
        'BCR_ABL':   np.random.choice([0,1], n_leukemia, p=[0.55, 0.45]),
        'TP53':      np.random.choice([0,1], n_leukemia, p=[0.50, 0.50]),
        'RUNX1':     np.random.choice([0,1], n_leukemia, p=[0.55, 0.45]),
        'DNMT3A':    np.random.choice([0,1], n_leukemia, p=[0.50, 0.50]),
        'NPM1':      np.random.choice([0,1], n_leukemia, p=[0.45, 0.55]),
        'CEBPA':     np.random.choice([0,1], n_leukemia, p=[0.60, 0.40]),
        'WT1':       np.random.choice([0,1], n_leukemia, p=[0.55, 0.45]),

        # Layer 2 — Gene expression levels 
        # High expression = gene overactive
        'FLT3_expression':  np.random.normal(8.5, 1.5, n_leukemia).clip(0, 15),
        'MYC_expression':   np.random.normal(9.2, 1.8, n_leukemia).clip(0, 15),
        'BCL2_expression':  np.random.normal(7.8, 1.2, n_leukemia).clip(0, 15),
        'HOXA9_expression': np.random.normal(8.1, 1.6, n_leukemia).clip(0, 15),
        'ERG_expression':   np.random.normal(7.5, 1.4, n_leukemia).clip(0, 15),

        # Layer 3 — Chromosomal abnormalities 
        'trisomy_8':    np.random.choice([0,1], n_leukemia, p=[0.75, 0.25]),
        'monosomy_7':   np.random.choice([0,1], n_leukemia, p=[0.80, 0.20]),
        'del_5q':       np.random.choice([0,1], n_leukemia, p=[0.82, 0.18]),
        'inv_16':       np.random.choice([0,1], n_leukemia, p=[0.85, 0.15]),
        't_8_21':       np.random.choice([0,1], n_leukemia, p=[0.88, 0.12]),

        # Layer 4 — Clinical markers 
        'wbc_count':        np.random.normal(85000, 20000, n_leukemia).clip(1000, 200000),
        'blast_percentage': np.random.normal(65, 20, n_leukemia).clip(20, 100),
        'ldh_level':        np.random.normal(850, 200, n_leukemia).clip(100, 2000),
        'hemoglobin':       np.random.normal(7.5, 1.5, n_leukemia).clip(3, 18),
        'platelet_count':   np.random.normal(50000, 15000, n_leukemia).clip(5000, 500000),

        # Layer 5 — Treatment response 
        'chemo_sensitivity':     np.random.normal(0.6, 0.2, n_leukemia).clip(0, 1),
        'relapse_risk':          np.random.normal(0.65, 0.2, n_leukemia).clip(0, 1),
        'mutation_burden':       np.random.normal(18, 5, n_leukemia).clip(0, 50),

        'has_leukemia': 1
    })

    
    # HEALTHY PATIENTS
    healthy = pd.DataFrame({

        'age':    np.random.randint(1, 18, n_healthy),
        'gender': np.random.choice(['M', 'F'], n_healthy),

        # Binary mutations — rare
        'FLT3_ITD':  np.random.choice([0,1], n_healthy, p=[0.99, 0.01]),
        'BCR_ABL':   np.random.choice([0,1], n_healthy, p=[0.99, 0.01]),
        'TP53':      np.random.choice([0,1], n_healthy, p=[0.97, 0.03]),
        'RUNX1':     np.random.choice([0,1], n_healthy, p=[0.98, 0.02]),
        'DNMT3A':    np.random.choice([0,1], n_healthy, p=[0.98, 0.02]),
        'NPM1':      np.random.choice([0,1], n_healthy, p=[0.99, 0.01]),
        'CEBPA':     np.random.choice([0,1], n_healthy, p=[0.99, 0.01]),
        'WT1':       np.random.choice([0,1], n_healthy, p=[0.99, 0.01]),

        # Gene expression — normal levels
        'FLT3_expression':  np.random.normal(3.2, 0.8, n_healthy).clip(0, 8),
        'MYC_expression':   np.random.normal(3.5, 0.9, n_healthy).clip(0, 8),
        'BCL2_expression':  np.random.normal(3.0, 0.7, n_healthy).clip(0, 8),
        'HOXA9_expression': np.random.normal(2.8, 0.6, n_healthy).clip(0, 8),
        'ERG_expression':   np.random.normal(3.1, 0.8, n_healthy).clip(0, 8),

        # Chromosomal — mostly normal
        'trisomy_8':    np.random.choice([0,1], n_healthy, p=[0.99, 0.01]),
        'monosomy_7':   np.random.choice([0,1], n_healthy, p=[0.99, 0.01]),
        'del_5q':       np.random.choice([0,1], n_healthy, p=[0.99, 0.01]),
        'inv_16':       np.random.choice([0,1], n_healthy, p=[0.99, 0.01]),
        't_8_21':       np.random.choice([0,1], n_healthy, p=[0.99, 0.01]),

        # Clinical markers — normal ranges
        'wbc_count':        np.random.normal(7500, 1500, n_healthy).clip(3000, 15000),
        'blast_percentage': np.random.normal(1.5, 0.8, n_healthy).clip(0, 5),
        'ldh_level':        np.random.normal(180, 40, n_healthy).clip(100, 300),
        'hemoglobin':       np.random.normal(13.5, 1.5, n_healthy).clip(8, 18),
        'platelet_count':   np.random.normal(250000, 50000, n_healthy).clip(100000, 500000),

        # Treatment response — N/A for healthy
        'chemo_sensitivity':     np.random.normal(0.5, 0.1, n_healthy).clip(0, 1),
        'relapse_risk':          np.random.normal(0.05, 0.02, n_healthy).clip(0, 0.2),
        'mutation_burden':       np.random.normal(1.5, 0.5, n_healthy).clip(0, 5),

        'has_leukemia': 0
    })

    # COMBINE
    df = pd.concat([leukemia, healthy], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total patients:     {len(df)}")
    print(f"Leukemia cases:     {df['has_leukemia'].sum()}")
    print(f"Healthy cases:      {(df['has_leukemia']==0).sum()}")
    print(f"Features per patient: {len(df.columns)}")
    print(f"\nFeature layers:")
    print(f"  Layer 1 — Binary mutations:      8 features")
    print(f"  Layer 2 — Gene expression:       5 features")
    print(f"  Layer 3 — Chromosomal:           5 features")
    print(f"  Layer 4 — Clinical markers:      5 features")
    print(f"  Layer 5 — Treatment response:    3 features")
    print(f"  Total:                          26 features")

    return df


if __name__ == "__main__":
    df = generate_v2_dataset(500)
    df.to_csv('genomic_v2_dataset.csv', index=False)
    print(f"\nSaved genomic_v2_dataset.csv")

    # stats
    leuk = df[df['has_leukemia']==1]
    healthy = df[df['has_leukemia']==0]

    print(f"\nKey differences leukemia vs healthy:")
    print(f"WBC count:    {leuk['wbc_count'].mean():.0f} vs {healthy['wbc_count'].mean():.0f}")
    print(f"Blast %:      {leuk['blast_percentage'].mean():.1f}% vs {healthy['blast_percentage'].mean():.1f}%")
    print(f"FLT3 expr:    {leuk['FLT3_expression'].mean():.1f} vs {healthy['FLT3_expression'].mean():.1f}")
    print(f"Mutation burden: {leuk['mutation_burden'].mean():.1f} vs {healthy['mutation_burden'].mean():.1f}")

    
    print("damn version 2 dataset ready!")
    