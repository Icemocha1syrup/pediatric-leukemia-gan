
# SynthPeds — Custom GAN from scratch
# Generator + Discriminator built with PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


# 1.GENERATOR
# Takes random noise → outputs fake patient

class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # Layer 1 — expand noise into hidden layer
            nn.Linear(noise_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Layer 2 — learn more complex patterns
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            # Layer 3 — refine patterns
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            # Output — produce a fake patient
            nn.Linear(128, output_dim),
            nn.Tanh()  # output between -1 and 1
        )

    def forward(self, noise):
        return self.model(noise)



# 2.DISCRIMINATOR
# Takes a patient record → real or fake

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Layer 1 — analyse the patient record
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # Layer 2 — look for suspicious patterns
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # Layer 3 — final judgement
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            # Output — probability of being REAL
            nn.Linear(128, 1),
            nn.Sigmoid()  # 0 = fake, 1 = real
        )

    def forward(self, x):
        return self.model(x)


# 3.DATA PREPARATION
# Convert CSV into numbers the GAN can learn from
def prepare_data(df):
    print("  Preparing data...")
    df_encoded = df.copy()
    encoders = {}
    scalers = {}

    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype.name == 'bool':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df_encoded[col] = scaler.fit_transform(df_encoded[[col]])
        scalers[col] = scaler

    data = torch.FloatTensor(df_encoded.values)
    print(f"  Data shape: {data.shape}")
    return data, encoders, scalers, df_encoded.columns.tolist()



# 4.TRAINING LOOP
# Generator and discriminator fight each other
def train_gan(df, epochs=200, batch_size=64, noise_dim=100, lr=0.0002):
    print("="*50)
    print("SynthPeds Custom GAN Training")
    print("="*50)

    # Prepare data
    real_data, encoders, scalers, columns = prepare_data(df)
    data_dim = real_data.shape[1]

    # Build generator and discriminator
    G = Generator(noise_dim, data_dim)
    D = Discriminator(data_dim)

    print(f"  Generator params:     {sum(p.numel() for p in G.parameters()):,}")
    print(f"  Discriminator params: {sum(p.numel() for p in D.parameters()):,}")
    print()

    # Optimizers — Adam works best for GANs
    g_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    g_losses = []
    d_losses = []

    print("Training...")
    print(f"{'Epoch':>6} | {'D Loss':>8} | {'G Loss':>8} | {'D(real)':>8} | {'D(fake)':>8}")
    print("-" * 50)

    for epoch in range(epochs):

        #  TRAIN DISCRIMINATOR 
        # real patients's output = 1
        # fake patients's output = 0 

        idx = torch.randint(0, len(real_data), (batch_size,))
        real_batch = real_data[idx]
        real_labels = torch.ones(batch_size, 1)

        noise = torch.randn(batch_size, noise_dim)
        fake_batch = G(noise).detach()
        fake_labels = torch.zeros(batch_size, 1)

        d_optimizer.zero_grad()
        d_real = D(real_batch)
        d_fake = D(fake_batch)
        d_loss_real = criterion(d_real, real_labels)
        d_loss_fake = criterion(d_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        d_optimizer.step()

        # TRAIN GENERATOR 
        # Generate fakes → fool discriminator 
        # The output should be 1
        noise = torch.randn(batch_size, noise_dim)
        fake_batch = G(noise)
        fool_labels = torch.ones(batch_size, 1)

        g_optimizer.zero_grad()
        d_fake_for_g = D(fake_batch)
        g_loss = criterion(d_fake_for_g, fool_labels)
        g_loss.backward()
        g_optimizer.step()

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())

        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"{epoch+1:>6} | {d_loss.item():>8.4f} | {g_loss.item():>8.4f} | {d_real.mean().item():>8.4f} | {d_fake.mean().item():>8.4f}")

    print()
    print("Training complete!")
    return G, D, scalers, encoders, columns, g_losses, d_losses



# 5. — GENERATE SYNTHETIC PATIENTS
# Use trained generator to mint fake kids
def generate_patients(G, scalers, encoders, columns, n_patients=1000, noise_dim=100):
    print(f"\nGenerating {n_patients} synthetic patients...")
    G.eval()
    with torch.no_grad():
        noise = torch.randn(n_patients, noise_dim)
        fake_scaled = G(noise).numpy()

    # Reverse the scaling back to original values
    fake_df = pd.DataFrame(fake_scaled, columns=columns)
    for col in columns:
        fake_df[col] = scalers[col].inverse_transform(fake_df[[col]])

    # Round integer columns
    for col in ['age', 'hospital_visits']:
        if col in fake_df.columns:
            fake_df[col] = fake_df[col].round().astype(int).clip(0, 18 if col == 'age' else 30)

    # Decode categorical columns
    for col, le in encoders.items():
        if col in fake_df.columns:
            fake_df[col] = fake_df[col].round().astype(int).clip(0, len(le.classes_) - 1)
            fake_df[col] = le.inverse_transform(fake_df[col])

    print(f"Generated {len(fake_df)} synthetic patients!")
    return fake_df


# 6. RUN EVERYTHING
if __name__ == "__main__":

    # Create demo dataset (replace with Synthea on hackathon day)
    print("Loading demo pediatric dataset...")
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'age':             np.random.randint(0, 18, n),
        'gender':          np.random.choice(['M', 'F'], n),
        'diagnosis':       np.random.choice(['diabetes', 'asthma', 'cancer', 'healthy'], n, p=[0.2, 0.3, 0.1, 0.4]),
        'glucose_level':   np.random.normal(120, 30, n).round(1),
        'bmi':             np.random.normal(18, 3, n).round(1),
        'hospital_visits': np.random.randint(0, 20, n),
        'on_medication':   np.random.choice([True, False], n),
    })
    print(f"Loaded {len(df)} real patient records\n")

    # Train
    G, D, scalers, encoders, columns, g_losses, d_losses = train_gan(
        df,
        epochs=200,
        batch_size=64,
        noise_dim=100
    )

    # Generate
    fake_df = generate_patients(G, scalers, encoders, columns, n_patients=1000)

    # Save
    fake_df.to_csv('gan_synthetic_patients.csv', index=False)

    # Compare real vs fake
    print("\nREAL DATA sample:")
    print(df.head(5).to_string(index=False))
    print("\nFAKE DATA sample:")
    print(fake_df.head(5).to_string(index=False))

    print("\n" + "="*50)
    print("SUCCESS — Custom GAN working!")
    print("File saved: gan_synthetic_patients.csv")
    print("="*50)