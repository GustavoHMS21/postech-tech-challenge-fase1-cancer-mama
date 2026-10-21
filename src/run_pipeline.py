"""Executa o pipeline completo em um comando: EDA -> treino -> avaliação -> explicação.

Por que existe? Para que README, Dockerfile e a verificação final tenham UM ponto de
entrada, em vez de quatro comandos separados. Não contém lógica: só chama os módulos.
"""
from src import eda, evaluate, explain, train

if __name__ == "__main__":
    eda.run_eda()
    print("EDA ok")
    print("Validação:\n", train.train_all().round(4))
    print("Teste:\n", evaluate.evaluate_all().round(4))
    explain.plot_feature_importance(); explain.plot_shap_summary(); explain.plot_shap_individual()
    print("Figuras em reports/figures/. Pipeline concluído.")
