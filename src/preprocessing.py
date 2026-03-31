import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(['PatientId', 'AppointmentID'], axis=1)

    # Convert target variable
    df['No-show'] = df['No-show'].map({'No': 1, 'Yes': 0})
    df.rename(columns={'No-show': 'FollowUp'}, inplace=True)

    # Convert dates
    df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
    df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

    # Feature engineering: waiting days
    df['WaitingDays'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days

    # Remove negative waiting days
    df = df[df['WaitingDays'] >= 0]

    # Encode Gender
    df['Gender'] = df['Gender'].map({'M': 1, 'F': 0})

    return df