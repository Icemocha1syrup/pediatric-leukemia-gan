import pandas as pd
import numpy as np
from gan import train_gan, generate_patients

# Load genomic dataset
df = pd.read_csv('genomic_dataset.csv')

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# Drop text columns — GAN needs numbers only
df_numeric = df.drop([
    'gender',
    'FLT3_sequence',
    'BCR_ABL_sequence',
    'TP53_sequence',
    'RUNX1_sequence',
    'NPM1_sequence',
    'DNMT3A_sequence',
    'CEBPA_sequence',
    'WT1_sequence'
], axis=1)

print("\nNumeric shape:", df_numeric.shape)
print("Columns kept:", df_numeric.columns.tolist())

# Train GAN on genomic data
print("\nTraining GAN on genomic data...")
G, D, scalers, encoders, columns, g_losses, d_losses = train_gan(
    df_numeric,
    epochs=1000,
    batch_size=32,
    noise_dim=128
)

# Generate 1000 synthetic genomic profiles
print("\nGenerating 1000 synthetic genomic profiles...")
fake_genomic = generate_patients(
    G, scalers, encoders, columns,
    n_patients=1000,
    noise_dim=128
)

# Round mutation columns to 0 or 1
mutation_cols = ['FLT3_ITD', 'BCR_ABL', 'TP53', 
                 'RUNX1', 'DNMT3A', 'NPM1', 
                 'CEBPA', 'WT1', 'has_leukemia']

for col in mutation_cols:
    if col in fake_genomic.columns:
        fake_genomic[col] = fake_genomic[col].round().astype(int).clip(0, 1)

# Round age and chromosome
fake_genomic['age'] = fake_genomic['age'].round().astype(int).clip(1, 18)
fake_genomic['chromosome'] = fake_genomic['chromosome'].round().astype(int).clip(1, 22)

print("\nMutation distribution in synthetic data:")
print(f"Has leukemia: {fake_genomic['has_leukemia'].sum()} / {len(fake_genomic)}")
print(f"FLT3_ITD:     {fake_genomic['FLT3_ITD'].sum()}")
print(f"BCR_ABL:      {fake_genomic['BCR_ABL'].sum()}")
print(f"TP53:         {fake_genomic['TP53'].sum()}")
print(f"RUNX1:        {fake_genomic['RUNX1'].sum()}")
print(f"DNMT3A:       {fake_genomic['DNMT3A'].sum()}")
print(f"NPM1:         {fake_genomic['NPM1'].sum()}")
print(f"CEBPA:        {fake_genomic['CEBPA'].sum()}")
print(f"WT1:          {fake_genomic['WT1'].sum()}")

# Real world frequencies from medical literature
real_frequencies = {
    'FLT3_ITD': 0.35,  # 35% of AML
    'BCR_ABL':  0.25,  # 25% of ALL
    'TP53':     0.20,  # 20% high risk
    'RUNX1':    0.25,  # 25% AML
    'DNMT3A':   0.22,  # 22% AML
    'NPM1':     0.30,  # 30% AML
    'CEBPA':    0.15,  # 15% AML
    'WT1':      0.18,  # 18% poor prognosis
}

print("\nApplying frequency correction...")
leukemia_mask = fake_genomic['has_leukemia'] == 1
leukemia_df = fake_genomic[leukemia_mask].copy()

for col, real_rate in real_frequencies.items():
    current_rate = leukemia_df[col].mean()
    print(f"{col}: current={current_rate:.2f} → target={real_rate:.2f}")

    if current_rate > real_rate:
        # Too many mutations — randomly remove some
        excess = current_rate - real_rate
        mutation_idx = leukemia_df[leukemia_df[col]==1].index
        n_remove = int(excess * len(leukemia_df))
        if n_remove > 0 and len(mutation_idx) > n_remove:
            remove_idx = np.random.choice(
                mutation_idx, n_remove, replace=False
            )
            fake_genomic.loc[remove_idx, col] = 0

    elif current_rate < real_rate:
        # Too few mutations — randomly add some
        deficit = real_rate - current_rate
        no_mutation_idx = leukemia_df[leukemia_df[col]==0].index
        n_add = int(deficit * len(leukemia_df))
        if n_add > 0 and len(no_mutation_idx) > n_add:
            add_idx = np.random.choice(
                no_mutation_idx, n_add, replace=False
            )
            fake_genomic.loc[add_idx, col] = 1

print("\nAfter correction:")
leukemia_corrected = fake_genomic[fake_genomic['has_leukemia']==1]
for col in real_frequencies.keys():
    print(f"{col}: {leukemia_corrected[col].mean():.2f} (target: {real_frequencies[col]})")
# Save
fake_genomic.to_csv('synthetic_genomes.csv', index=False)
print("\nCleaned synthetic genomes:")
print(fake_genomic.head(5).to_string(index=False))

