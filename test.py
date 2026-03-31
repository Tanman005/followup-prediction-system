from src.preprocessing import load_data, preprocess_data
from src.model import train_models

df = load_data("data/appointments.csv")
df = preprocess_data(df)

train_models(df)