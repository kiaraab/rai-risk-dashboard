# Mitigated Fairness Audit with UCI Adult Dataset
# Dependencies: fairlearn, pandas, scikit-learn, matplotlib

from fairlearn.metrics import MetricFrame, selection_rate, demographic_parity_difference
from fairlearn.metrics import equalized_odds_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load and preprocess data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
cols = ["age","workclass","fnlwgt","education","education-num","marital-status",
        "occupation","relationship","race","sex","capital-gain","capital-loss",
        "hours-per-week","native-country","income"]
data = pd.read_csv(url, names=cols, na_values=" ?", skipinitialspace=True)
data = data.dropna()
data['income'] = (data['income'] == '>50K').astype(int)

le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])  # 0=Female,1=Male
X = pd.get_dummies(data.drop(columns=['income']), drop_first=True)
y = data['income']
A = data['sex']  # sensitive feature

# 2. Split and scale
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    X, y, A, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Base model
base_model = LogisticRegression(max_iter=5000)
base_model.fit(X_train, y_train)
y_pred_base = base_model.predict(X_test)

# 4. Fairness metrics before mitigation
mf_base = MetricFrame(
    metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
    y_true=y_test, y_pred=y_pred_base, sensitive_features=A_test)
print("Before mitigation:")
print(mf_base.by_group)
print("Demographic parity gap:", demographic_parity_difference(y_test, y_pred_base, sensitive_features=A_test))
print("Equalized odds difference:", equalized_odds_difference(y_test, y_pred_base, sensitive_features=A_test))

# 5. Mitigate using Exponentiated Gradient (Demographic Parity)
mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(max_iter=5000),
    constraints=DemographicParity(),
    eps=0.02
)
mitigator.fit(X_train, y_train, sensitive_features=A_train)
pred_mitigated = mitigator.predict(X_test)

# 6. Metrics after mitigation
mf_mitigated = MetricFrame(
    metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
    y_true=y_test, y_pred=pred_mitigated, sensitive_features=A_test)
print("After mitigation:")
print(mf_mitigated.by_group)
print("Demographic parity gap after:", demographic_parity_difference(y_test, pred_mitigated, sensitive_features=A_test))
print("Equalized odds difference after:", equalized_odds_difference(y_test, pred_mitigated, sensitive_features=A_test))

# 7. Plot comparison
import numpy as np

groups = ['Female','Male']
acc_base = mf_base.by_group['accuracy'].values
acc_mit = mf_mitigated.by_group['accuracy'].values
sr_base = mf_base.by_group['selection_rate'].values
sr_mit = mf_mitigated.by_group['selection_rate'].values

x = np.arange(len(groups))
width = 0.35
fig, ax = plt.subplots()
ax.bar(x - width/2, sr_base, width, label='Base')
ax.bar(x + width/2, sr_mit, width, label='Mitigated')
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel('Selection Rate')
ax.set_title('Selection Rate by Gender: Base vs Mitigated')
ax.legend()
plt.tight_layout()
plt.show()
