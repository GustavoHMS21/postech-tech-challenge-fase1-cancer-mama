"""Executa todas as etapas em sequência: EDA, treino, avaliação e interpretação."""
from src import eda, evaluate, explain, train

if __name__ == "__main__":
    eda.run_eda()
    print("EDA concluída")
    print("Validação:\n", train.train_all().round(4))
    print("Teste:\n", evaluate.evaluate_all().round(4))
    print("Erros no teste:\n", evaluate.error_analysis())
    evaluate.plot_confusion_matrices()
    explain.plot_feature_importance()
    explain.plot_shap_summary()
    explain.plot_shap_individual()
    print("Figuras salvas em reports/figures/")
