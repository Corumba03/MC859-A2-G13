import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURAÇÃO ---
HEURISTIC_DIRS = {
    'GRASP Clássico': 'logs_classic',
    'GRASP Reativo': 'logs_reactive',
    'Random plus Greedy': 'logs_random_plus_greed'
}

# Lista de padrões para o TAMANHO DA INSTÂNCIA (do nome do arquivo)
SIZE_PATTERNS_FILENAME = [
    re.compile(r'exp_n(\d+)'),
    re.compile(r'gen_(\d+)'),
    re.compile(r'instance_(\d+)')
]

# Padrões para extrair o CUSTO de DENTRO do arquivo
# Padrão para a linha de solução final (maior prioridade)
FINAL_COST_PATTERN = re.compile(r'Final Solution:.*?cost=\[(-?[\d.]+)\]')
# Padrão para as linhas de iteração (usado como fallback)
ITER_COST_PATTERN = re.compile(r'\(Iter\. \d+\) BestSol.*?cost=\[(-?[\d.]+)\]')
# --- FIM DA CONFIGURAÇÃO ---


def extract_quality_data_from_logs(log_dir):
    """
    Extrai dados de qualidade. Prioriza o custo da 'Final Solution'.
    Se não encontrar, usa o custo da última iteração registrada.
    """
    results = []
    if not os.path.isdir(log_dir):
        print(f"❌ Erro: O diretório '{log_dir}' não foi encontrado.")
        return []

    for filename in os.listdir(log_dir):
        if filename.endswith(".log"):
            instance_size = None
            for pattern in SIZE_PATTERNS_FILENAME:
                match = pattern.search(filename)
                if match:
                    instance_size = int(match.group(1))
                    break
            
            if instance_size is None:
                print(f"⚠️  Aviso: Nenhum padrão de nome de arquivo correspondeu a '{filename}'. Pulando.")
                continue

            filepath = os.path.join(log_dir, filename)
            final_cost = None
            last_known_cost = None # Variável para guardar o último custo de iteração

            with open(filepath, 'r') as f:
                for line in f:
                    # Verifica se é uma linha de iteração e atualiza o último custo conhecido
                    iter_match = ITER_COST_PATTERN.search(line)
                    if iter_match:
                        last_known_cost = float(iter_match.group(1))

                    # Verifica se é a linha de solução final (que tem prioridade)
                    final_match = FINAL_COST_PATTERN.search(line)
                    if final_match:
                        final_cost = float(final_match.group(1))
                        break # Encontrou o definitivo, pode parar
            
            # Decide qual custo usar
            best_cost_found = None
            if final_cost is not None:
                best_cost_found = final_cost # Prioridade 1: Custo final
            elif last_known_cost is not None:
                best_cost_found = last_known_cost # Prioridade 2: Custo da última iteração
            
            if best_cost_found is not None:
                print(f"✔️  Arquivo '{filename}': Tamanho={instance_size}, Custo={best_cost_found}")
                results.append({'instance_size': instance_size, 'best_cost': best_cost_found})
            else:
                print(f"❌ Aviso: Não foi possível encontrar nenhum custo em '{filename}'. Pulando.")

    return results


def plot_quality_comparison_chart(data):
    """
    Esta função permanece a mesma.
    """
    if not data:
        print("❌ Nenhum dado de qualidade para plotar.")
        return

    df = pd.DataFrame(data)
    df = df.sort_values(by=['heuristic', 'instance_size'])
    print("\n📊 Gerando o gráfico de qualidade da solução...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df, x='instance_size', y='best_cost', hue='heuristic', marker='o', linewidth=2.5, errorbar=('ci', 95))
    sns.scatterplot(data=df, x='instance_size', y='best_cost', hue='heuristic', alpha=0.4, legend=False)
    plt.title('Comparativo de Qualidade da Solução Final', fontsize=18, fontweight='bold')
    plt.xlabel('Tamanho da Instância (n)', fontsize=14)
    plt.ylabel('Custo da Melhor Solução (Média)', fontsize=14)
    plt.legend(title='Heurística', fontsize=12)
    plt.grid(True, which="both", ls="--")
    output_filename = 'grafico_comparativo_qualidade.png'
    plt.savefig(output_filename)
    print(f"✅ Gráfico salvo como '{output_filename}'")
    plt.show()


if __name__ == '__main__':
    all_results = []
    
    for heuristic_name, dir_path in HEURISTIC_DIRS.items():
        print(f"\n--- Processando heurística: {heuristic_name} ---")
        results_from_dir = extract_quality_data_from_logs(dir_path)
        for result in results_from_dir:
            result['heuristic'] = heuristic_name
        all_results.extend(results_from_dir)

    plot_quality_comparison_chart(all_results)