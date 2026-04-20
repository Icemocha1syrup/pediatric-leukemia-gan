import pandas as pd
import numpy as np
from Bio.Seq import Seq
import warnings
warnings.filterwarnings('ignore')

# LEUKEMIA GENOMIC MUTATIONS
# Based on real clinical research:
# - FLT3-ITD: 25% of pediatric AML
# - BCR-ABL:  Philadelphia chromosome ALL
# - TP53:     High risk leukemia
# - RUNX1:    AML predisposition
# - DNMT3A:   Epigenesis mutation
# - NPM1:     Good prognosis AML
# - CEBPA:    Balletic good prognosis
# - WT1:      Poor prognosis marker


def generate_leukemia_dataset(n_patients=500):
    print("Generating pediatric leukemia genomic dataset...")
    np.random.seed(42)

    
    # LEUKEMIA PATIENTS (50% of population)
    n_leukemia = int(n_patients * 0.5)
    n_healthy   = n_patients - n_leukemia

    # Leukemia patients — mutations follow
    # real clinical frequency distributions
    leukemia = pd.DataFrame({
        # Demographics
        'age':    np.random.randint(1, 18, n_leukemia),
        'gender': np.random.choice(['M', 'F'], n_leukemia),

        # Key mutations — HIGH frequency in leukemia
        'FLT3_ITD':  np.random.choice([0,1], n_leukemia, p=[0.45, 0.55]),  # 35% AML
        'BCR_ABL':   np.random.choice([0,1], n_leukemia, p=[0.55, 0.45]),  # 25% ALL
        'TP53':      np.random.choice([0,1], n_leukemia, p=[0.50, 0.50]),  # 20% high risk
        'RUNX1':     np.random.choice([0,1], n_leukemia, p=[0.55, 0.45]),  # 25% AML
        'DNMT3A':    np.random.choice([0,1], n_leukemia, p=[0.50, 0.50]),  # 22% AML
        'NPM1':      np.random.choice([0,1], n_leukemia, p=[0.45, 0.55]),  # 30% AML
        'CEBPA':     np.random.choice([0,1], n_leukemia, p=[0.60, 0.40]),  # 15% AML
        'WT1':       np.random.choice([0,1], n_leukemia, p=[0.55, 0.45]),  # 18% poor prognosis

        # DNA sequence features
        'chromosome': np.random.choice([1,2,4,5,7,8,9,11,13,17,21,22], n_leukemia),
        'mutation_burden': np.random.normal(15, 5, n_leukemia).round(1).clip(0, 50),

        # Label
        'has_leukemia': 1
    })

    
    # HEALTHY CHILDREN — rare/no mutations
    healthy = pd.DataFrame({
        'age':    np.random.randint(1, 18, n_healthy),
        'gender': np.random.choice(['M', 'F'], n_healthy),

        # Same mutations but very LOW frequency
        'FLT3_ITD':  np.random.choice([0,1], n_healthy, p=[0.99, 0.01]),
        'BCR_ABL':   np.random.choice([0,1], n_healthy, p=[0.99, 0.01]),
        'TP53':      np.random.choice([0,1], n_healthy, p=[0.97, 0.03]),
        'RUNX1':     np.random.choice([0,1], n_healthy, p=[0.98, 0.02]),
        'DNMT3A':    np.random.choice([0,1], n_healthy, p=[0.98, 0.02]),
        'NPM1':      np.random.choice([0,1], n_healthy, p=[0.99, 0.01]),
        'CEBPA':     np.random.choice([0,1], n_healthy, p=[0.99, 0.01]),
        'WT1':       np.random.choice([0,1], n_healthy, p=[0.99, 0.01]),

        'chromosome': np.random.choice([1,2,4,5,7,8,9,11,13,17,21,22], n_healthy),
        'mutation_burden': np.random.normal(1, 0.5, n_healthy).round(1).clip(0, 5),

        'has_leukemia': 0
    })

   
    # COMBINE AND SHUFFLE
    df = pd.concat([leukemia, healthy], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total patients:    {len(df)}")
    print(f"Leukemia cases:    {df['has_leukemia'].sum()}")
    print(f"Healthy cases:     {(df['has_leukemia']==0).sum()}")
    print(f"Mutations tracked: FLT3-ITD, BCR-ABL, TP53, RUNX1, DNMT3A, NPM1, CEBPA, WT1")
    print()
    print("Sample data:")
    print(df.head(5).to_string(index=False))

    return df



# DNA SEQUENCE GENERATOR
# Uses Biopython to create realistic
# synthetic DNA sequences around mutations

def generate_dna_sequence(mutation_present, gene_name):
    # Base sequences for each gene
    base_sequences = {
    'FLT3_ITD': 'ATGCAGTGGTTGGCAGCAATGAAAGCAATTTTCGTACTGAAAGGTTGTGGTTCAGCAGAC',
    'BCR_ABL':  'ATGCAGAGTATCTGGCCAGCTTTTGAAGCAGTTCGGGAAGTGGCAGAGCTTGAAGACTTC',
    'TP53':     'ATGGAGGAGCCGCAGTCAGATCCTAGCGTTGAATGAGCAGGTCAGCTTTGAGGTGCATG',
    'RUNX1':    'ATGCGTATCCCCGTAGATGCCAGCAGCACTCCATATGGATCGAGTGGATGGGCCCAGCAA',
    'NPM1':     'ATGAAAAAGAAAGAAATCGAAAACCAGAGGAAGAGAATTTCAGAAGAGAAGAAACGAAAG',
    'DNMT3A':   'ATGCCAGCAGAAGCAGAGCTGGAGAAGGAGAAGCAGCAGCTGCAGAAGGAGCAGCAGAAG',
    'CEBPA':    'ATGCAGCAGCAGCAGCAGCAGCAGCAGCAGCAGGCGGAGCAGCAGCAGCAGCAGCAGCAG',
    'WT1':      'ATGGGCGACCGCGGTGAGCCCGGCGCCTACCCGTACCCGCAGCCGCAGCCGCAGCAGCCG',
    }

    base = base_sequences.get(gene_name, 'ATCGATCGATCGATCGATCG' * 3)
    seq = Seq(base)

    if mutation_present:
        # Insert ITD duplication for FLT3
        if gene_name == 'FLT3_ITD':
            insert = str(seq[11:35])  # duplicate region
            mutated = str(seq[:11]) + insert + str(seq[11:])
            return mutated[:60]
        # Point mutation for others
        seq_list = list(str(seq))
        pos = np.random.randint(10, len(seq_list)-10)
        mutations = {'A':'T', 'T':'A', 'C':'G', 'G':'C'}
        seq_list[pos] = mutations.get(seq_list[pos], 'A')
        return ''.join(seq_list)

    return str(seq)



# RUN
if __name__ == "__main__":
    # Generate dataset
    df = generate_leukemia_dataset(500)

    # Add DNA sequences for key mutations
print("\nGenerating DNA sequences for all mutations...")
df['FLT3_sequence']  = df['FLT3_ITD'].apply(
    lambda x: generate_dna_sequence(x, 'FLT3_ITD'))

df['BCR_ABL_sequence'] = df['BCR_ABL'].apply(
    lambda x: generate_dna_sequence(x, 'BCR_ABL'))

df['TP53_sequence']  = df['TP53'].apply(
    lambda x: generate_dna_sequence(x, 'TP53'))

df['RUNX1_sequence'] = df['RUNX1'].apply(
    lambda x: generate_dna_sequence(x, 'RUNX1'))

df['NPM1_sequence']  = df['NPM1'].apply(
    lambda x: generate_dna_sequence(x, 'NPM1'))

df['DNMT3A_sequence'] = df['DNMT3A'].apply(
    lambda x: generate_dna_sequence(x, 'DNMT3A'))

df['CEBPA_sequence'] = df['CEBPA'].apply(
    lambda x: generate_dna_sequence(x, 'CEBPA'))

df['WT1_sequence']   = df['WT1'].apply(
    lambda x: generate_dna_sequence(x, 'WT1'))

    # Save
df.to_csv('genomic_dataset.csv', index=False)
print(f"\nSaved genomic_dataset.csv")
print("\n" + "="*50)
print("SUCCESS — Genomic dataset ready!")
print("="*50)
