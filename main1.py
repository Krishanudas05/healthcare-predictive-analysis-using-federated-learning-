import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the stroke dataset
dataset_path = 'healthcare-dataset-stroke-data.csv'
df = pd.read_csv(dataset_path)

# Drop unnecessary columns (if any) and encode labels if needed
df = df.drop(columns=['id'])
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
df['work_type'] = df['work_type'].map({'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4})
df['Residence_type'] = df['Residence_type'].map({'Urban': 0, 'Rural': 1})
df['smoking_status'] = df['smoking_status'].map({'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3})
df['stroke'] = df['stroke'].astype(int)

# Separate features and labels
X = df.drop(columns=['stroke'])
y = df['stroke']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert data to dictionary format for federated learning
clients_data = {}
for i in range(len(X_train)):
    client_name = f'client_{i}'
    clients_data[client_name] = {'features': X_train[i, :], 'label': y_train.iloc[i]}

# Define an enhanced global model architecture
global_model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Federated Averaging (FedAvg) algorithm
def federated_averaging(global_model, clients_data, num_rounds, aggregation_rate):
    global_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    for round_num in range(num_rounds):
        selected_clients = np.random.choice(list(clients_data.keys()), size=aggregation_rate, replace=False)
        client_models = {client: tf.keras.models.clone_model(global_model) for client in selected_clients}

        for client in selected_clients:
            local_model = client_models[client]
            client_data = clients_data[client]['features']
            label = clients_data[client]['label']

            # Train local model
            local_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            local_model.fit(np.expand_dims(client_data, axis=0), np.array([label]), epochs=50, verbose=0)  # Increase epochs

            # Update local model in the dictionary
            client_models[client] = local_model

        # Aggregate local model updates using simple averaging
        global_model_weights = global_model.get_weights()
        for i, layer_weights in enumerate(zip(*[model.get_weights() for model in client_models.values()])):
            global_model_weights[i] = np.mean(layer_weights, axis=0)

        global_model.set_weights(global_model_weights)

        # Evaluate the model on the training set (optional)
        _, train_accuracy = global_model.evaluate(X_train, y_train, verbose=0)
        print(f'Round {round_num + 1}/{num_rounds} - Training Accuracy: {train_accuracy * 100:.2f}%')

    return global_model

# Run Federated Learning with increased training rounds and epochs
global_model = federated_averaging(global_model, clients_data, num_rounds=30, aggregation_rate=5)

# Evaluate the final model on the test set
test_loss, test_acc = global_model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_acc * 100:.2f}%')

# Perform predictive analysis on the dataset
predictions = global_model.predict(X_test)
rounded_predictions = np.round(predictions).flatten()

# Display some results
results = pd.DataFrame({'Actual': y_test.values, 'Predicted': rounded_predictions})
print(results.head(10))

