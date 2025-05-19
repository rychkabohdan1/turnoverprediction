import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class TurnoverPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1  # Use all available cores for faster training
        )
        self.scaler = StandardScaler()
        self.department_encoding = {
            'Engineering': 1,
            'Sales': 2,
            'Marketing': 3,
            'HR': 4,
            'Finance': 5,
            'Operations': 6,
            'IT': 7
        }
        # New set of features - removed age, years_of_experience
        # Added time_since_promotion, manager_rating, work_flexibility, team_size
        self.feature_names = [
            'age',
            'salary',
            'department',
            'performance_rating',
            'work_hours',
            'projects_completed',
            'training_hours'
        ]
        self._train_model()
        self.feature_names = [
            'age', 'salary', 'department',
            'performance_rating', 'work_hours', 'projects_completed', 'training_hours'
        ]

    def _create_training_data(self):
        # Create comprehensive training data with all 10 features
        # Format: [age, years_exp, salary, dept, perf_rating, work_hrs, projects, training_hrs, promotion_time, team_size]
        
        # First, let's generate a large, balanced training set with more realistic variation
        X_dummy = []
        y_dummy = []
        
        # For each department
        for dept in range(1, 7):  # 1=Engineering, 2=Sales, 3=Marketing, 4=HR, 5=Finance, 6=Operations
            # Generate low risk examples (more data for more accurate predictions)
            for _ in range(45):  # Increased from 15 to 45 examples per department
                age = np.random.randint(25, 40)
                years_exp = np.random.randint(1, 10)
                salary = np.random.randint(35000, 80000)
                perf_rating = np.random.uniform(3.8, 5.0)
                work_hrs = np.random.randint(35, 45)
                projects = np.random.randint(3, 8)
                training_hrs = np.random.randint(20, 50)
                promotion_time = np.random.randint(3, 12)
                team_size = np.random.randint(4, 12)
                
                X_dummy.append([age, years_exp, salary, dept, perf_rating, work_hrs, projects, training_hrs, promotion_time, team_size])
                y_dummy.append(0)  # 0 means stay (low risk)
            
            # Generate medium risk examples
            for _ in range(45):  # Increased from 15 to 45 examples per department
                age = np.random.randint(25, 55)
                years_exp = np.random.randint(1, 20)
                salary = np.random.randint(25000, 70000)
                perf_rating = np.random.uniform(2.5, 4.0)
                work_hrs = np.random.randint(40, 50)
                projects = np.random.randint(2, 6)
                training_hrs = np.random.randint(10, 30)
                promotion_time = np.random.randint(12, 24)
                team_size = np.random.randint(3, 15)
                
                X_dummy.append([age, years_exp, salary, dept, perf_rating, work_hrs, projects, training_hrs, promotion_time, team_size])
                y_dummy.append(np.random.choice([0, 1], p=[0.7, 0.3]))  # 30% chance of leaving (medium risk)
            
            # Generate high risk examples
            for _ in range(45):  # Increased from 15 to 45 examples per department
                age = np.random.randint(30, 60)
                years_exp = np.random.randint(1, 25)
                salary = np.random.randint(20000, 60000)
                perf_rating = np.random.uniform(1.0, 3.5)
                work_hrs = np.random.randint(45, 60)
                projects = np.random.randint(1, 4)
                training_hrs = np.random.randint(0, 15)
                promotion_time = np.random.randint(18, 36)
                team_size = np.random.randint(2, 20)
                
                X_dummy.append([age, years_exp, salary, dept, perf_rating, work_hrs, projects, training_hrs, promotion_time, team_size])
                y_dummy.append(np.random.choice([0, 1], p=[0.3, 0.7]))  # 70% chance of leaving (high risk)
        
        # Generate specific cases to reinforce key retention factors
        
        # Low salary is a strong predictor of turnover
        for _ in range(50):  # Increased from 20 to 50 examples
            dept = np.random.randint(1, 7)
            X_dummy.append([
                np.random.randint(25, 50),
                np.random.randint(1, 15),
                np.random.randint(18000, 25000),  # Very low salary
                dept,
                np.random.uniform(2.0, 4.5),
                np.random.randint(35, 55),
                np.random.randint(1, 7),
                np.random.randint(5, 40),
                np.random.randint(6, 30),
                np.random.randint(3, 15)
            ])
            y_dummy.append(1)  # Likely to leave
        
        # High work hours correlate with turnover
        for _ in range(50):  # Increased from 20 to 50 examples
            dept = np.random.randint(1, 7)
            X_dummy.append([
                np.random.randint(25, 50),
                np.random.randint(1, 15),
                np.random.randint(30000, 70000),
                dept,
                np.random.uniform(2.0, 4.5),
                np.random.randint(50, 60),  # High work hours
                np.random.randint(1, 7),
                np.random.randint(5, 40),
                np.random.randint(6, 30),
                np.random.randint(3, 15)
            ])
            y_dummy.append(1)  # Likely to leave
        
        # Low performance rating correlates with turnover
        for _ in range(50):  # Increased from 20 to 50 examples
            dept = np.random.randint(1, 7)
            X_dummy.append([
                np.random.randint(25, 50),
                np.random.randint(1, 15),
                np.random.randint(30000, 70000),
                dept,
                np.random.uniform(1.0, 2.5),  # Low performance
                np.random.randint(35, 55),
                np.random.randint(1, 7),
                np.random.randint(5, 40),
                np.random.randint(6, 30),
                np.random.randint(3, 15)
            ])
            y_dummy.append(1)  # Likely to leave
        
        # Low training hours correlate with turnover
        for _ in range(50):  # Increased from 20 to 50 examples
            dept = np.random.randint(1, 7)
            X_dummy.append([
                np.random.randint(25, 50),
                np.random.randint(1, 15),
                np.random.randint(30000, 70000),
                dept,
                np.random.uniform(2.0, 4.5),
                np.random.randint(35, 55),
                np.random.randint(1, 7),
                np.random.randint(0, 10),  # Very low training
                np.random.randint(6, 30),
                np.random.randint(3, 15)
            ])
            y_dummy.append(1)  # Likely to leave
        
        # High salary and good conditions generally retain employees
        for _ in range(50):  # Increased from 20 to 50 examples
            dept = np.random.randint(1, 7)
            X_dummy.append([
                np.random.randint(25, 50),
                np.random.randint(1, 15),
                np.random.randint(70000, 100000),  # High salary
                dept,
                np.random.uniform(4.0, 5.0),       # High performance
                np.random.randint(35, 45),         # Reasonable hours
                np.random.randint(3, 7),
                np.random.randint(25, 50),         # Good training
                np.random.randint(3, 12),          # Recent promotion
                np.random.randint(5, 12)
            ])
            y_dummy.append(0)  # Likely to stay
        
        # Add more nuanced cases for better model training
        
        # Long promotion gap but otherwise good conditions
        for _ in range(30):
            dept = np.random.randint(1, 7)
            X_dummy.append([
                np.random.randint(30, 45),
                np.random.randint(5, 15),
                np.random.randint(50000, 80000),
                dept,
                np.random.uniform(3.5, 4.8),
                np.random.randint(35, 45),
                np.random.randint(3, 7),
                np.random.randint(20, 40),
                np.random.randint(24, 48),  # Long time since promotion
                np.random.randint(5, 10)
            ])
            y_dummy.append(np.random.choice([0, 1], p=[0.5, 0.5]))  # 50-50 chance
            
        # Young employees with low experience but high performance
        for _ in range(30):
            dept = np.random.randint(1, 7)
            X_dummy.append([
                np.random.randint(22, 28),  # Young age
                np.random.randint(0, 3),    # Low experience
                np.random.randint(30000, 55000),
                dept,
                np.random.uniform(4.0, 5.0),  # High performance
                np.random.randint(40, 50),
                np.random.randint(2, 5),
                np.random.randint(15, 40),
                np.random.randint(0, 6),     # Recent hire/promotion
                np.random.randint(4, 10)
            ])
            y_dummy.append(np.random.choice([0, 1], p=[0.6, 0.4]))  # 40% turnover risk
            
        # Senior employees with high experience
        for _ in range(30):
            dept = np.random.randint(1, 7)
            X_dummy.append([
                np.random.randint(45, 60),   # Older age
                np.random.randint(15, 30),   # High experience
                np.random.randint(60000, 90000),
                dept,
                np.random.uniform(3.0, 4.5),
                np.random.randint(35, 50),
                np.random.randint(3, 8),
                np.random.randint(10, 30),
                np.random.randint(12, 60),   # Potentially long time since promotion
                np.random.randint(5, 15)
            ])
            y_dummy.append(np.random.choice([0, 1], p=[0.7, 0.3]))  # 30% turnover risk
            
        # Large team size with varied conditions
        for _ in range(30):
            dept = np.random.randint(1, 7)
            X_dummy.append([
                np.random.randint(25, 50),
                np.random.randint(3, 20),
                np.random.randint(40000, 75000),
                dept,
                np.random.uniform(2.5, 4.5),
                np.random.randint(35, 55),
                np.random.randint(2, 6),
                np.random.randint(10, 40),
                np.random.randint(6, 36),
                np.random.randint(15, 30)    # Large team
            ])
            y_dummy.append(np.random.choice([0, 1], p=[0.6, 0.4]))  # 40% turnover risk
            
        # Small team size with varied conditions
        for _ in range(30):
            dept = np.random.randint(1, 7)
            X_dummy.append([
                np.random.randint(25, 50),
                np.random.randint(3, 20),
                np.random.randint(40000, 75000),
                dept,
                np.random.uniform(2.5, 4.5),
                np.random.randint(35, 55),
                np.random.randint(2, 6),
                np.random.randint(10, 40),
                np.random.randint(6, 36),
                np.random.randint(1, 4)      # Small team
            ])
            y_dummy.append(np.random.choice([0, 1], p=[0.5, 0.5]))  # 50% turnover risk
            
        # Convert to numpy arrays
        X_dummy = np.array(X_dummy)
        y_dummy = np.array(y_dummy)
        
        print(f"Created training dataset with {len(X_dummy)} examples")
        
        return X_dummy, y_dummy

    def _train_model(self):
        X_dummy, y_dummy = self._create_training_data()
        X_scaled = self.scaler.fit_transform(X_dummy)
        self.model.fit(X_scaled, y_dummy)

    def predict(self, employee_data):
        # Extract features
        features = np.array([
            employee_data.age,
            employee_data.years_of_experience,
            employee_data.salary,
            self.department_encoding.get(employee_data.department, 1),  # Default to 1 (Engineering) if department not found
            employee_data.performance_rating,
            employee_data.work_hours,
            employee_data.projects_completed,
            employee_data.training_hours,
            employee_data.time_since_promotion,
            employee_data.team_size
        ]).reshape(1, -1)

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict probability
        probability = self.model.predict_proba(features_scaled)[0][1]

        # Determine risk level
        if probability < 0.3:
            risk_level = "low"
        elif probability < 0.6:
            risk_level = "medium"
        else:
            risk_level = "high"

        return {
            "turnover_probability": float(probability),
            "risk_level": risk_level
        }

    def analyze_feature_importance(self):
        """Return the feature importance from the trained model"""
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model doesn't have feature importances")

        feature_importance = self.model.feature_importances_
        return {name: importance for name, importance in zip(self.feature_names, feature_importance)}

    def predict_with_feature_impact(self, employee_data):
        """
        Predict turnover probability and analyze how changes to different features 
        would impact the prediction.
        
        Returns a dict with:
        - prediction: the base prediction
        - feature_impacts: list of feature impacts and recommendations
        """
        # Get base prediction
        base_prediction = self.predict(employee_data)
        base_probability = base_prediction["turnover_probability"]
        
        # Extract original features for reference (keeping all 10 for prediction compatibility)
        original_features = np.array([
            employee_data.age,
            employee_data.years_of_experience,
            employee_data.salary,
            self.department_encoding.get(employee_data.department, 1),  # Default to 1 (Engineering) if department not found
            employee_data.performance_rating,
            employee_data.work_hours,
            employee_data.projects_completed,
            employee_data.training_hours,
            employee_data.time_since_promotion,
            employee_data.team_size
        ]).reshape(1, -1)
        
        # Scale the original features
        original_scaled = self.scaler.transform(original_features)
        
        # Define feature names for reference - only including the ones we'll actually modify
        feature_names = [
            "age", 
            "years_of_experience",
            "salary",
            "department",
            "performance_rating",
            "work_hours",
            "projects_completed",
            "training_hours",
            "time_since_promotion",
            "team_size"
        ]
        
        # Get feature importance from the model
        feature_importances = dict(zip(feature_names, self.model.feature_importances_))
        
        # Define factor magnitudes that represent meaningful changes
        factor_magnitudes = {
            "salary": 10000,               # $10k increase
            "performance_rating": 0.5,     # 0.5 point increase on 5-point scale
            "work_hours": -5,              # 5 hour reduction
            "training_hours": 15           # 15 additional training hours
        }
        
        # Define factor descriptions for better UI display
        factor_descriptions = {
            "salary": "Compensation Level",
            "performance_rating": "Performance Rating",
            "work_hours": "Weekly Work Hours",
            "training_hours": "Annual Training Hours"
        }
        
        # Define potential improvements
        feature_improvements = []
        
        # Test improvements for each modifiable factor
        for factor, magnitude in factor_magnitudes.items():
            # Find index of the factor in feature array
            idx = feature_names.index(factor)
            
            # Create a modified version of the features
            modified_features = original_features.copy()
            
            # Apply the magnitude change
            if factor == "salary":
                improved_value = max(modified_features[0, idx] * 1.1, modified_features[0, idx] + magnitude)
                # Round to nearest 1000
                improved_value = round(improved_value / 1000) * 1000
                modified_features[0, idx] = improved_value
            elif factor == "work_hours":
                # Don't go below 35 hours
                improved_value = max(35, modified_features[0, idx] + magnitude)
                modified_features[0, idx] = improved_value
            else:
                # Generic change
                current_value = modified_features[0, idx]
                improved_value = min(5.0, current_value + magnitude) if factor == "performance_rating" else current_value + magnitude
                modified_features[0, idx] = improved_value
            
            # Scale the modified features
            modified_scaled = self.scaler.transform(modified_features)
            
            # Get new prediction
            new_prob = self.model.predict_proba(modified_scaled)[0][1]
            
            # Calculate impact
            impact = new_prob - base_probability
            
            # Calculate percent change for better understanding
            percent_change = (1 - (new_prob / base_probability)) * 100 if base_probability > 0 else 0
            
            # Format value display for UI
            if factor == "salary":
                current_display = f"${int(original_features[0, idx]):,}"
                improved_display = f"${int(improved_value):,}"
            elif factor == "performance_rating":
                current_display = f"{original_features[0, idx]:.1f} / 5.0"
                improved_display = f"{improved_value:.1f} / 5.0"
            elif factor == "work_hours":
                current_display = f"{int(original_features[0, idx])} hrs/week"
                improved_display = f"{int(improved_value)} hrs/week"
            elif factor == "training_hours":
                current_display = f"{int(original_features[0, idx])} hrs/year"
                improved_display = f"{int(improved_value)} hrs/year"
            else:
                current_display = f"{original_features[0, idx]}"
                improved_display = f"{improved_value}"
            
            # Calculate risk level changes
            current_risk = "High Risk" if base_probability >= 0.6 else "Medium Risk" if base_probability >= 0.3 else "Low Risk"
            new_risk = "High Risk" if new_prob >= 0.6 else "Medium Risk" if new_prob >= 0.3 else "Low Risk"
            risk_change = current_risk != new_risk
            
            # Format action text
            if factor == "salary":
                action = f"Increase salary from {current_display} to {improved_display}"
                explanation = "Competitive compensation is one of the strongest retention factors"
            elif factor == "performance_rating":
                action = f"Improve performance rating from {current_display} to {improved_display}"
                explanation = "Higher job satisfaction correlates with improved performance"
            elif factor == "work_hours":
                action = f"Reduce weekly work hours from {current_display} to {improved_display}"
                explanation = "Better work-life balance reduces burnout and turnover risk"
            elif factor == "training_hours":
                action = f"Increase training hours from {current_display} to {improved_display}"
                explanation = "Professional development opportunities increase employee loyalty"
                
            # Add to improvements list (even if impact is 0)
            feature_improvements.append({
                "feature": factor,
                "display_name": factor_descriptions[factor],
                "current_value": float(original_features[0, idx]),
                "recommended_value": float(improved_value),
                "current_display": current_display,
                "improved_display": improved_display,
                "impact": float(impact),
                "percent_change": float(percent_change),
                "importance": float(feature_importances[factor]),
                "action": action,
                "explanation": explanation,
                "risk_change": risk_change,
                "new_probability": float(new_prob),
                "new_risk_level": new_risk
            })
        
        # Sort improvements by impact (most negative impact first = most helpful)
        feature_improvements.sort(key=lambda x: x["impact"])
        
        # Filter to include only recommendations that help or are negligible
        helpful_improvements = [imp for imp in feature_improvements if imp["impact"] <= 0.01]
        
        # If no helpful improvements were found, add the least harmful one
        if not helpful_improvements and feature_improvements:
            helpful_improvements = [feature_improvements[0]]
            
        # Calculate the biggest potential benefit
        if helpful_improvements:
            max_improvement = min([imp["impact"] for imp in helpful_improvements])
            max_percent = max([imp["percent_change"] for imp in helpful_improvements])
        else:
            max_improvement = 0
            max_percent = 0
            
        # Calculate new probability after all improvements
        best_new_probability = max(0.01, base_probability + max_improvement)
        
        # Determine new risk level after all improvements
        if best_new_probability < 0.3:
            best_new_risk_level = "Low Risk"
        elif best_new_probability < 0.6:
            best_new_risk_level = "Medium Risk"
        else:
            best_new_risk_level = "High Risk"
        
        return {
            "prediction": base_prediction,
            "feature_impacts": helpful_improvements,
            "max_improvement": float(max_improvement),
            "max_percent_improvement": float(max_percent),
            "estimated_new_probability": float(best_new_probability),
            "estimated_new_risk_level": best_new_risk_level
        }

    def save_model(self, path="models"):
        os.makedirs(path, exist_ok=True)
        joblib.dump(self.model, os.path.join(path, "turnover_model.joblib"))
        joblib.dump(self.scaler, os.path.join(path, "scaler.joblib"))

    def load_model(self, path="models"):
        self.model = joblib.load(os.path.join(path, "turnover_model.joblib"))
        self.scaler = joblib.load(os.path.join(path, "scaler.joblib")) 