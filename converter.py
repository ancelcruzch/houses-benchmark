# convertir_csv_a_pt.py
import torch
import pandas as pd

# Carga CSV
df = pd.read_csv("dataset/data1/HouseTS_log.csv")

# Puedes transformar el DataFrame en un tensor (por ejemplo, solo valores numéricos)
tensor = torch.tensor(df.select_dtypes(include='number').values, dtype=torch.float)

# Guarda como .pt
torch.save(tensor, "dataset/data1/HouseTS_tensor.pt")


