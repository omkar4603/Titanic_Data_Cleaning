Task 1: Data Cleaning & Preprocessing – Titanic Dataset

Objective
The goal of this task is to **learn how to clean and prepare raw data** for machine learning models.  
The Titanic dataset is used to demonstrate handling missing data, encoding categorical variables, feature scaling, and outlier removal.

---

Tools & Libraries Used
- **Python**
- **Pandas** – for data manipulation and cleaning  
- **NumPy** – for numerical operations  
- **Matplotlib / Seaborn** – for visualization  
- **Scikit-Learn** – for encoding and feature scaling  

---

Dataset InformationDataset:** `titanic.csv`  Description:** Contains passenger data from the Titanic disaster.

| Column | Description |
|:-------|:-------------|
| `PassengerId` | Unique ID of passenger |
| `Survived` | Survival status (0 = No, 1 = Yes) |
| `Pclass` | Passenger class (1, 2, or 3) |
| `Name` | Name of the passenger |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | # of siblings/spouses aboard |
| `Parch` | # of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Ticket fare |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

Steps Performed

1. Data Exploration
- Loaded the dataset using `pandas.read_csv()`
- Displayed basic info, data types, and summary statistics
- Identified missing values
2. Handling Missing Values
- Filled missing `Age` values using **median**
- Filled missing `Embarked` values using **mode**
- Dropped the `Cabin` column due to high number of missing values

3. Encoding Categorical Variables
- Converted `Sex` into numerical form using **LabelEncoder**
- Applied **One-Hot Encoding** for the `Embarked` column

4. Feature Scaling
- Scaled numerical columns `Age` and `Fare` using **StandardScaler**

5. Outlier Detection & Removal
- Visualized `Fare` outliers using a **boxplot**
- Removed extreme outliers using the **IQR method**

6. Exporting Cleaned Data
- Saved the final cleaned dataset as  
  **`cleaned_titanic.csv`**

---

Visualization Example
A boxplot was used to detect outliers in the `Fare` column.

```python
sns.boxplot(x=df['Fare'])
plt.title("Fare Boxplot (Before Outlier Removal)")
plt.show()
