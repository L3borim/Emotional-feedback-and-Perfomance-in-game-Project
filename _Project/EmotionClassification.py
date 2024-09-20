import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.ensemble import (AdaBoostClassifier,
                              ExtraTreesClassifier,
                              RandomForestClassifier)

from sklearn.metrics import (accuracy_score, classification_report, f1_score, precision_score, recall_score)

from sklearn.model_selection import (cross_val_predict, cross_val_score,
                                     train_test_split)

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier


# Carregar as tabelas de emoções e estatísticas
#emotion_data = pd.read_csv('emotion_detection_results.csv')
#performance_data = pd.read_csv('game_performance_statistics.csv')

# Mesclar os dados pelas colunas de tempo (aproximando o tempo mais próximo)
merged_data = pd.merge_asof(emotion_data.sort_values('Time'), performance_data.sort_values('Time'), on='Time')

# Codificar as emoções como números inteiros
label_encoder = LabelEncoder()
merged_data['Emotion'] = label_encoder.fit_transform(merged_data['Emotion'])

# Definir as features e o target (exemplo: performance)
# Supondo que a coluna 'Performance' seja algo como 'alta', 'média', 'baixa'
X = merged_data[['Emotion', 'Kills', 'Deaths', 'Assists']]  # Features
y = merged_data['Performance']  # Target de classificação (pode ser uma métrica numérica ou categórica)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

# Definir o modelo Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinar o modelo
rf_model.fit(X_train, y_train)

# Fazer previsões
y_pred = rf_model.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Accuracy: {accuracy:.2f}")

# Imprimir o relatório de classificação
print(classification_report(y_test, y_pred))

# Definir o modelo XGBoost
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Treinar o modelo
xgb_model.fit(X_train, y_train)

# Fazer previsões
y_pred_xgb = xgb_model.predict(X_test)

# Avaliar o modelo
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")

# Obter as importâncias das features
feature_importances = rf_model.feature_importances_

# Visualizar as importâncias
sns.barplot(x=feature_importances, y=X.columns)
plt.title("Feature Importance in Random Forest")
plt.show()

xgb.plot_importance(xgb_model)
plt.show()