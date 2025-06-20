import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score as accuracy , precision_score as precision , recall_score as recall , f1_score 
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

class ClassifierPipeline:
    def __init__(self,df,feature_cols,target_col):
        self.df = df.copy()
        self.features = feature_cols
        self.target_col = target_col
       
        self.models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Logistic Regression": LogisticRegression(),
            "Naive Bayes": GaussianNB(),
            "SVM": SVC(probability=True),  # Enable predict_proba for future use
        }
        self.metrics_summary=[]

    def prepare_data(self):
        X = self.df[self.features]
        y= self.df[self.target_col]
        X_Scaled = StandardScaler().fit_transform(X)
        self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(X_Scaled,y,test_size=0.2,random_state=42,shuffle=True)

    def execute(self):
        self.prepare_data()
        for name,model in self.models.items():
            model.fit(self.X_train,self.y_train)
            y_pred = model.predict(self.X_test)

            #calculations
            acc = accuracy(self.y_test,y_pred)
            prec = precision(self.y_test,y_pred,average='weighted')
            rec = recall(self.y_test,y_pred,average='weighted')
            f1 = f1_score(self.y_test,y_pred,average='weighted')

            self.metrics_summary.append({
                "Model":name,
                "Accuracy": acc,
                "Precision":prec,
                "Recall":rec,
                "F1-score":f1
            })

        return pd.DataFrame(self.metrics_summary)
        
