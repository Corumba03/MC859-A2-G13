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

# PADRÕES DE EXTRAÇÃO
# AGORA TEMOS UMA LISTA DE PADRÕES PARA O NOME DO ARQUIVO
SIZE_PATTERNS_FILENAME = [
    re.compile(r'exp_n(\d+)'),        # Padrão 1: para 'exp_n25.log'
    re.compile(r'gen_(\d+)'),         # Padrão 2: para 'gen_100_k4...'
    re.compile(r'instance_(\d+)')    # Padrão 3: para 'instance_14.log'
]

# Padrões para extrair o TEMPO de DENTRO do arquivo (inalterados)
TIME_PATTERN = re.compile(r'Finished in ([\d.]+) seconds')
TIMEOUT_PATTERN = re.compile(r'Execution timed out after ([\d.]+) seconds')
# --- FIM DA CONFIGURAÇÃO ---


def extract_data_from_logs(log_dir):
    """
    CORRIGIDO: Tenta múltiplos padrões para encontrar o tamanho da instância
    no nome do arquivo e extrai o tempo de execução de dentro do arquivo.
    """
    results = []
    if not os.path.isdir(log_dir):
        print(f"❌ Erro: O diretório '{log_dir}' não foi encontrado.")
        return []

    for filename in os.listdir(log_dir):
        if filename.endswith(".log"):
            # 1. Tenta cada padrão da lista até encontrar o tamanho da instância
            instance_size = None
            for pattern in SIZE_PATTERNS_FILENAME:
                match = pattern.search(filename)
                if match:
                    instance_size = int(match.group(1))
                    break  # Encontrou um padrão correspondente, para de procurar

            if instance_size is None:
                print(f"⚠️  Aviso: Nenhum padrão de nome de arquivo correspondeu a '{filename}'. Pulando.")
                continue

            # 2. Abre o arquivo para encontrar o TEMPO DE EXECUÇÃO
            filepath = os.path.join(log_dir, filename)
            execution_time = None
            timed_out = False

            with open(filepath, 'r') as f:
                for line in f:
                    timeout_match = TIMEOUT_PATTERN.search(line)
                    if timeout_match:
                        execution_time = float(timeout_match.group(1))
                        timed_out = True
                        break

                    time_match = TIME_PATTERN.search(line)
                    if time_match:
                        execution_time = float(time_match.group(1))
                        break
            
            if execution_time is not None:
                if timed_out:
                    print(f"⚠️  TIMEOUT: '{filename}' -> Tamanho={instance_size}, Tempo={execution_time:.2f}s")
                else:
                    print(f"✔️  Arquivo '{filename}': Tamanho={instance_size}, Tempo={execution_time:.2f}s")
                results.append({'instance_size': instance_size, 'execution_time': execution_time})
            else:
                print(f"❌ Aviso: Não foi possível encontrar o tempo de execução em '{filename}'. Pulando.")

    return results


def plot_comparison_chart(data):
    """
    Esta função permanece a mesma.
    """
    if not data:
        print("❌ Nenhum dado para plotar.")
        return
    df = pd.DataFrame(data)
    df = df.sort_values(by=['heuristic', 'instance_size'])
    print("\n📊 Gerando o gráfico comparativo...")
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=df, x='instance_size', y='execution_time', hue='heuristic', marker='o', linewidth=2.5, errorbar=('ci', 95))
    sns.scatterplot(data=df, x='instance_size', y='execution_time', hue='heuristic', alpha=0.4, legend=False)
    plt.title('Comparativo de Tempo de Execução entre Heurísticas GRASP', fontsize=18, fontweight='bold')
    plt.xlabel('Tamanho da Instância (n)', fontsize=14)
    plt.ylabel('Tempo de Execução (segundos) - Escala Logarítmica', fontsize=14)
    plt.legend(title='Heurística', fontsize=12)
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    output_filename = 'grafico_comparativo_tempo.png'
    plt.savefig(output_filename)
    print(f"✅ Gráfico salvo como '{output_filename}'")
    plt.show()


if __name__ == '__main__':
    all_results = []
    
    for heuristic_name, dir_path in HEURISTIC_DIRS.items():
        print(f"\n--- Processando heurística: {heuristic_name} ---")
        results_from_dir = extract_data_from_logs(dir_path)
        for result in results_from_dir:
            result['heuristic'] = heuristic_name
        all_results.extend(results_from_dir)

    plot_comparison_chart(all_results)