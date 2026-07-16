import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import mlflow
import mlflow.sklearn

# 1. Configurar MLflow
# Creamos o seleccionamos un "Experimento" en MLflow para agrupar nuestras ejecuciones
mlflow.set_experiment("Deteccion_Fraude_Tarjetas")

# Activamos el registro automático para Scikit-Learn
mlflow.sklearn.autolog()

# 2. Cargar y limpiar tus datos usando la ruta real de descargas
path_csv = r"C:\Users\Usuario\Downloads\creditcard.csv\creditcard.csv"
print(f"Cargando datos desde: {path_csv}...")

df = pd.read_csv(path_csv)
df.columns = df.columns.str.replace('"', '').str.strip()
df['Class'] = df['Class'].astype(str).str.replace('"', '').str.strip().astype(int)

X = df.drop(columns=['Class', 'Time'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 3. Iniciar el registro del "Run" en MLflow
with mlflow.start_run(run_name="RandomForest_Base") as run:
    
    # Definimos el modelo
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Validación cruzada
    print("\nEjecutando Validación Cruzada Estratificada de 5 Folds...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)
    mean_f1 = cv_scores.mean()
    
    # Registramos manualmente la métrica de validación cruzada que nos interesa
    mlflow.log_metric("cv_mean_f1", mean_f1)
    
    # Entrenamos el modelo final
    print("\nEntrenando modelo final...")
    model.fit(X_train, y_train)
    
    # Evaluación en Test (Examen final)
    y_pred = model.predict(X_test)
    test_score = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", test_score)
    
    # Imprimimos en consola los detalles del reporte clásico para tener feedback visual
    print("\n--- REPORTE EN EL SET DE TEST ---")
    print(f"Accuracy global: {test_score:.4f}")
    print(classification_report(y_test, y_pred))
    
    # Guardamos el modelo en local como siempre
    joblib.dump(model, 'modelo_fraude.pkl')
    
    print(f"\n¡Entrenamiento completado! CV Mean F1: {mean_f1:.4f}")