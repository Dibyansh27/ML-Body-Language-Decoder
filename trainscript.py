# model_training.py
# Body Language Model Training and Evaluation

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class BodyLanguageModel:
    def __init__(self, data_path='Dataset/body_language_data.csv'):
        self.data_path = data_path
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare the dataset"""
        print("Loading dataset...")
        try:
            self.df = pd.read_csv(self.data_path)
            print(f"Dataset loaded with shape: {self.df.shape}")
            print(f"Classes: {self.df['class'].unique()}")
            print(f"Class distribution:\n{self.df['class'].value_counts()}")
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
    
    def exploratory_data_analysis(self):
        """Perform exploratory data analysis"""
        print("\n=== Exploratory Data Analysis ===")
        
        # Basic statistics
        print("\nDataset Info:")
        print(f"Shape: {self.df.shape}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        
        # Class distribution plot
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        self.df['class'].value_counts().plot(kind='bar')
        plt.title('Class Distribution')
        plt.xlabel('Classes')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Feature statistics
        plt.subplot(1, 2, 2)
        feature_cols = [col for col in self.df.columns if col != 'class']
        self.df[feature_cols].mean().hist(bins=50)
        plt.title('Feature Distribution')
        plt.xlabel('Feature Values')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('eda_analysis.png')
        plt.show()
        
        return True
    
    def prepare_data(self):
        """Prepare data for training"""
        print("\nPreparing data for training...")
        
        # Separate features and target
        feature_cols = [col for col in self.df.columns if col != 'class']
        X = self.df[feature_cols].values
        y = self.df['class'].values
        
        # Handle any missing values
        X = np.nan_to_num(X)
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        
        return True
    
    def train_models(self):
        """Train multiple models and compare performance"""
        print("\n=== Training Models ===")
        
        # Define models
        models = {
            'Logistic Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(random_state=42, max_iter=1000))
            ]),
            'Random Forest': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
            ]),
            'Gradient Boosting': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
            ]),
            'SVM': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(kernel='rbf', random_state=42, probability=True))
            ])
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)
            
            # Calculate accuracies
            train_acc = accuracy_score(self.y_train, train_pred)
            test_acc = accuracy_score(self.y_test, test_pred)
            
            results[name] = {
                'model': model,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'predictions': test_pred
            }
            
            print(f"{name} - Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
        
        self.models = results
        return results
    
    def evaluate_models(self):
        """Evaluate all models and select the best one"""
        print("\n=== Model Evaluation ===")
        
        # Compare accuracies
        model_comparison = pd.DataFrame({
            'Model': list(self.models.keys()),
            'Train Accuracy': [self.models[name]['train_accuracy'] for name in self.models.keys()],
            'Test Accuracy': [self.models[name]['test_accuracy'] for name in self.models.keys()]
        })
        
        print("\nModel Comparison:")
        print(model_comparison.to_string(index=False))
        
        # Select best model based on test accuracy and avoiding overfitting
        best_idx = model_comparison['Test Accuracy'].idxmax()
        self.best_model_name = model_comparison.iloc[best_idx]['Model']
        self.best_model = self.models[self.best_model_name]['model']
        
        print(f"\nBest Model: {self.best_model_name}")
        print(f"Test Accuracy: {model_comparison.iloc[best_idx]['Test Accuracy']:.4f}")
        
        # Detailed evaluation of best model
        best_predictions = self.models[self.best_model_name]['predictions']
        
        print(f"\nDetailed Classification Report for {self.best_model_name}:")
        print(classification_report(self.y_test, best_predictions))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, best_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=np.unique(self.y_test), 
                   yticklabels=np.unique(self.y_test))
        plt.title(f'Confusion Matrix - {self.best_model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        return model_comparison
    
    def save_model(self):
        """Save the best model"""
        print(f"\nSaving the best model: {self.best_model_name}")
        
        # Create Models directory
        os.makedirs('Models', exist_ok=True)
        
        # Save the model
        model_path = 'Models/body_language_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        print(f"Model saved to {model_path}")
        
        # Save model info
        model_info = {
            'model_name': self.best_model_name,
            'test_accuracy': self.models[self.best_model_name]['test_accuracy'],
            'classes': list(np.unique(self.y_test))
        }
        
        info_path = 'Models/model_info.pkl'
        with open(info_path, 'wb') as f:
            pickle.dump(model_info, f)
        
        print(f"Model info saved to {info_path}")
        
        return model_path
    
    def run_complete_training(self):
        """Run the complete training pipeline"""
        print("=== Body Language Model Training Pipeline ===")
        
        # Load data
        if not self.load_data():
            return False
        
        # EDA
        self.exploratory_data_analysis()
        
        # Prepare data
        if not self.prepare_data():
            return False
        
        # Train models
        results = self.train_models()
        
        # Evaluate models
        comparison = self.evaluate_models()
        
        # Save best model
        model_path = self.save_model()
        
        print("\n=== Training Complete ===")
        print(f"Best model: {self.best_model_name}")
        print(f"Model saved to: {model_path}")
        
        return True

def main():
    """Main function to run the training pipeline"""
    trainer = BodyLanguageModel()
    success = trainer.run_complete_training()
    
    if success:
        print("\nTraining completed successfully!")
    else:
        print("\nTraining failed. Please check your dataset.")

if __name__ == "__main__":
    main()