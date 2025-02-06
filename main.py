import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import logging

class HumidityPredictor:
    def __init__(self):
        """Initialize the HumidityPredictor with models and configuration."""
        # Initialize models
        self.models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(kernel='rbf', probability=True, random_state=42),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Logistic Regression': LogisticRegression(random_state=42)
        }
        
        # Initialize scaler
        self.scaler = StandardScaler()
        
        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load the weather dataset."""
        try:
            data = pd.read_csv(file_path)
            logging.info(f"Data loaded successfully with shape: {data.shape}")
            return data
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data including handling missing values and outliers."""
        # Remove unnecessary columns
        data = data.drop(['Unnamed: 11'], axis=1, errors='ignore')
        
        # Handle missing values
        for column in data.select_dtypes(include=[np.number]):
            data[column].fillna(data[column].median(), inplace=True)
        
        # Handle outliers using IQR
        for column in data.select_dtypes(include=[np.number]):
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[column] = np.clip(data[column], lower_bound, upper_bound)
        
        logging.info("Data preprocessing completed")
        return data

    def create_target(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Create features and target variables."""
        # Create binary target variable
        data['humidity_class'] = np.where(data['relative_humidity_3pm'] < 25, 0, 1)
        
        # Split features and target
        X = data.drop(['relative_humidity_3pm', 'humidity_class'], axis=1)
        y = data['humidity_class']
        
        logging.info(f"Target variable created. Class distribution:\n{y.value_counts()}")
        return X, y

    def train_models(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train all models and return their performance metrics."""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        for name, model in self.models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_scores': cv_scores,
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred),
                'predictions': y_pred,
                'true_values': y_test
            }
            
            logging.info(f"\n{name} Results:")
            logging.info(f"Accuracy: {accuracy:.4f}")
            logging.info(f"CV Scores: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results

    def plot_results(self, results: Dict):
        """Plot performance metrics for all models."""
        # Model Comparison Plot
        plt.figure(figsize=(12, 6))
        accuracies = [results[model]['accuracy'] for model in results.keys()]
        cv_scores = [results[model]['cv_scores'].mean() for model in results.keys()]
        
        x = np.arange(len(self.models))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Test Accuracy')
        plt.bar(x + width/2, cv_scores, width, label='CV Accuracy')
        
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(x, self.models.keys(), rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Confusion Matrices
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (name, result) in enumerate(results.items()):
            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', ax=axes[idx])
            axes[idx].set_title(f'{name} Confusion Matrix')
        
        plt.tight_layout()
        plt.show()

    def predict_future_humidity(self, X_new: pd.DataFrame, model_name: str = 'Random Forest') -> pd.DataFrame:
        """Predict humidity levels for new data."""
        # Scale the new data
        X_scaled = self.scaler.transform(X_new)
        
        # Get the selected model
        model = self.models[model_name]
        
        # Make predictions
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'Predicted_Class': predictions,
            'Probability_Low_Humidity': probabilities[:, 0],
            'Probability_High_Humidity': probabilities[:, 1]
        })
        
        results['Interpretation'] = results['Predicted_Class'].map({
            0: 'Low Humidity (<25%)',
            1: 'High Humidity (≥25%)'
        })
        
        return results

def main():
    # Initialize predictor
    predictor = HumidityPredictor()
    
    try:
        # Load and preprocess data
        data = predictor.load_data('rishi\Minor\weather.csv')
        processed_data = predictor.preprocess_data(data)
        
        # Create features and target
        X, y = predictor.create_target(processed_data)
        
        # Train models and get results
        results = predictor.train_models(X, y)
        
        # Plot results
        predictor.plot_results(results)
        
        # Example of future prediction
        print("\nExample prediction for new data:")
        new_data = X.head()  # Using first few rows as example
        predictions = predictor.predict_future_humidity(new_data)
        print(predictions)
        
    except Exception as e:
        logging.error(f"Error in main pipeline: {str(e)}")
        raise

main()