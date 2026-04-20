import pandas as pd
import numpy as np
from gan import train_gan, generate_patients

print("Loading V2 dataset...")
df = pd.read_csv('genomic_v2_dataset.csv')

# Drop text columns only
drop_cols = [col for col in ['gender'] if col in df.columns]
df_numeric = df.drop(drop_cols, axis=1)

print(f"Shape: {df_numeric.shape}")
print(f"Features: {df_numeric.columns.tolist()}")

# Train GAN
print("\nTraining V2 GAN...")
G, D, scalers, encoders, columns, g_losses, d_losses = train_gan(
    df_numeric,
    epochs=1000,
    batch_size=32,
    noise_dim=128
)

# Generate
print("\nGenerating 1000 V2 synthetic patients...")
fake_df = generate_patients(
    G, scalers, encoders, columns,
    n_patients=1000,
    noise_dim=128
)

# Round binary columns
binary_cols = ['FLT3_ITD', 'BCR_ABL', 'TP53', 'RUNX1',
               'DNMT3A', 'NPM1', 'CEBPA', 'WT1',
               'trisomy_8', 'monosomy_7', 'del_5q',
               'inv_16', 't_8_21', 'has_leukemia']
for col in binary_cols:
    if col in fake_df.columns:
        fake_df[col] = fake_df[col].round().astype(int).clip(0, 1)

if 'age' in fake_df.columns:
    fake_df['age'] = fake_df['age'].round().astype(int).clip(1, 18)

# Save
fake_df.to_csv('synthetic_genomes_v2.csv', index=False)

# Compare
leuk = fake_df[fake_df['has_leukemia']==1]
healthy = fake_df[fake_df['has_leukemia']==0]

print(f"\nResults:")
print(f"Total generated:  {len(fake_df)}")
print(f"Leukemia cases:   {len(leuk)}")
print(f"Healthy cases:    {len(healthy)}")
print(f"\nKey stats:")
print(f"WBC:     {leuk['wbc_count'].mean():.0f} vs {healthy['wbc_count'].mean():.0f}")
print(f"Blast %: {leuk['blast_percentage'].mean():.1f}% vs {healthy['blast_percentage'].mean():.1f}%")
print(f"FLT3:    {leuk['FLT3_expression'].mean():.1f} vs {healthy['FLT3_expression'].mean():.1f}")

print("\n" + "="*50)
print("SUCCESS — V2 synthetic genomes ready!")
print("="*50)