import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import glob

# Função auxiliar para demonstrar métricas dos modelos de classificação
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # Mudança aqui
    recall = recall_score(y_test, y_pred, average='weighted')        # Mudança aqui
    f1 = f1_score(y_test, y_pred, average='weighted')                # Mudança aqui

    print(f"{model_name} Metrics:")
    print(f"Acurácia: {accuracy:.4f}")
    print(f"Precisão: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    return accuracy, precision, recall, f1


EMOCOES_POSSIVEIS = ['anger', 'sadness', 'fear', 'surprise', 'happiness', 'neutral']

# Caminhos para os arquivos
arquivos_emocoes = glob.glob('./detection-results/*.csv')
arquivos_estatisticas = glob.glob('./game-stats/*.csv')

# Lista para armazenar todos os rounds
todos_os_rounds = []

# Processar cada par de arquivos
for emocoes_path, stats_path in zip(arquivos_emocoes, arquivos_estatisticas):
    emocoes_df = pd.read_csv(emocoes_path)
    emocoes_df['Time'] = emocoes_df['Time'].astype(float)
    estatisticas_df = pd.read_csv(stats_path, delimiter=';')

    for _, round_data in estatisticas_df.iterrows():
        start_time = round_data['Start_time']
        end_time = round_data['End_time']

        emocoes_round = emocoes_df[(emocoes_df['Time'] >= start_time) & (emocoes_df['Time'] < end_time)]
        emocoes_contagem = emocoes_round['Emotion'].value_counts().to_dict()
        emocoes_contagem.pop('neutral', None)

        round_info = round_data.to_dict()
        round_info.update({em: emocoes_contagem.get(em, 0) for em in EMOCOES_POSSIVEIS})
        
        if emocoes_contagem:
            round_info['emotion_class'] = max(emocoes_contagem, key=emocoes_contagem.get)
        else:
            round_info['emotion_class'] = 'neutral'

        todos_os_rounds.append(round_info)

# Criar DataFrame
analise_df = pd.DataFrame(todos_os_rounds)
analise_df = analise_df.fillna(0)

# Definir features e alvo
X = analise_df[['Kills', 'Deaths', 'Assists', 'Score', 'Round_result'] + EMOCOES_POSSIVEIS]
y = analise_df['emotion_class']

# Dividir dados
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Normalizar
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Dicionário de modelos
modelos = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=12),
    "SVM": SVC(kernel='linear', random_state=12),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(random_state=12),
    "Decision Tree": DecisionTreeClassifier(random_state=12)
}

# Treinar e avaliar cada modelo
for nome, modelo in modelos.items():
    # Avaliação
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    accuracy, precision, recall, f1 = evaluate_model(modelo, X_test, y_test, nome)
    print('\n')
