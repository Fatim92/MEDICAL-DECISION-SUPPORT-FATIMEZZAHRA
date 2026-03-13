## optimization function for Memory usage problem✅
import pandas as pd
import matplotlib as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
# Lecture des Données:

df=pd.read_csv("data_path") # read a csv dataset

#Exploration et Nettoyage des Données✅:
df.shape # return how much row and columns are there
df.head() #  return the first '5' rows
df.info() # prints a concise summary of a DataFrame to the console

## Missing Values✅
df.isna().sum()

 ## Distribution of classes✅
r=df["NObeyesdad"]              
r.value_counts(normalize=True)

## String problem✅
data=df
binary_classification_columns=["family_history_with_overweight","FAVC","SMOKE","SCC"]
for col in binary_classification_columns:
  data[col]=data[col].map({"yes":1,"no":0})
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})

## non-binary categorical problem ✅
le=LabelEncoder()
data["CAEC"]=le.fit_transform(df["CAEC"])
data["CALC"]=le.fit_transform(df["CALC"])
data["MTRANS"]=le.fit_transform(df["MTRANS"])
data["NObeyesdad"]=le.fit_transform(df["NObeyesdad"])

# check for anything unrealistic numerical columns like Age, Weight, Height ...ect✅
data[["Age","Height","Weight","FCVC","NCP","CH2O","FAF","TUE"]].describe()

## Correlation analysis✅
data.corr()

##Bonus Create a column that contain BMI Metric✅
data["BmiMetric"]=data["Weight"]/(data["Height"])**2   # BMI metric formula
##Correlation Heat Map✅
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Heatmap')
plt.show()

##loading dataset function ✅
def load_data(filepath="data_path", target_col="NObeyesdad"):
    df = pd.read_csv(filepath)
    df = df.reset_index(drop=True)
    X = df.drop(columns=[target_col])
    y = df[target_col].copy()
    y=y.values.ravel()
    return X, y

## optimization function for Memory usage problem✅
def optimize_memory(data): 
    for col in data.columns:
        if data[col].dtype == "float64":
            data[col] = data[col].astype("float32")
        elif data[col].dtype == "int64":
            data[col] = data[col].astype("int32")
    return data
## for more info visite notebooks/eda.ipynb