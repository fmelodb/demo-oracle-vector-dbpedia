import numpy as np
import oracledb
import time
import threading
from typing import List, Tuple, Dict
from dataclasses import dataclass
import argparse
from dotenv import load_dotenv
import os
import pandas as pd

#oracledb.init_oracle_client(lib_dir=r"D:\\instantclient_23_9") # change

# env
load_dotenv()

@dataclass
class QueryResult:
    """Armazena resultado de uma query individual"""
    query_id: int
    latency: float  # em segundos
    retrieved_ids: List[int]


def read_query_vectors_csv(filename: str, expected_dim: int = 1536) -> np.ndarray:
    """
    Lê arquivo CSV com vetores de consulta e retorna array numpy.
    Ignora a primeira coluna e valida se todos os vetores têm a dimensão esperada.
    Exclui os vetores que não têm a dimensão correta.

    Args:
        filename: caminho do arquivo CSV
        expected_dim: dimensão esperada dos vetores (default: 1536)

    Returns:
        np.ndarray: array 2D com shape (n_vectors, dimension)
    """
    df = pd.read_csv(filename)
    initial_count = len(df)

    # Remove a primeira coluna e converte para array float32
    vectors = df.iloc[:, 1:].values.astype(np.float32)

    # Valida dimensão de cada vetor
    valid_vectors = []
    excluded_vectors = []  # Armazena vetores excluídos completos

    for i, vec in enumerate(vectors):
        if len(vec) == expected_dim:
            valid_vectors.append(vec)
        else:
            # Armazena índice, dimensão e vetor completo
            if len(excluded_vectors) < 3:  # Apenas os primeiros 3
                excluded_vectors.append((i, len(vec), vec))

    vectors = np.array(valid_vectors, dtype=np.float32)

    # Log de validação
    excluded_count = len(vectors) - len(valid_vectors) + len(excluded_vectors)
    if excluded_count > 0:
        print(f"⚠ Validação de dimensão:")
        print(f"  - Total carregado: {initial_count}")
        print(f"  - Excluídos (dim != {expected_dim}): {initial_count - len(vectors)}")
        print(f"  - Mantidos: {len(vectors)}")
        if excluded_vectors:
            print(f"\n  Primeiros vetores excluídos:")
            for idx, dim, vec in excluded_vectors:
                print(f"    Vetor #{idx}: {dim} dimensões")
                print(f"      Valores: {vec}")

    if len(vectors) == 0:
        raise ValueError(f"Nenhum vetor válido encontrado com {expected_dim} dimensões")

    return vectors


def connect_database() -> oracledb.Connection:
    """
    Cria e retorna conexão com Oracle Database.

    Returns:
        oracledb.Connection: conexão ativa com o banco
    """

    username = os.getenv("DB_USER")
    password = os.getenv("DB_PASS")
    dsn = os.getenv("DB_URL")

    connection = oracledb.connect(
        user=username,
        password=password,
        dsn=dsn
    )

    return connection


def vector_search(connection: oracledb.Connection, query_vector: np.ndarray, top_k: int) -> Tuple[List[int], float]:
    """
    Executa busca vetorial no Oracle AI Vector Search.

    Args:
        connection: conexão com Oracle Database
        query_vector: vetor de consulta
        top_k: número de vizinhos mais próximos a retornar

    Returns:
        Tuple[List[int], float]: lista de IDs encontrados e tempo de execução
    """
    cursor = connection.cursor()

    # Converte vetor para formato Oracle
    vec_list = query_vector.tolist()
    vec_str = f"[{','.join(map(str, vec_list))}]"

    # Mede tempo de execução
    start_time = time.time()

    # Query de busca vetorial usando VECTOR_DISTANCE
    cursor.execute(f"""
        SELECT /*+ NO_RESULT_CACHE */ ID, TEXT
        FROM VECTOR_TABLE
        ORDER BY VECTOR_DISTANCE(EMBEDDING, TO_VECTOR(:vec), COSINE)
        FETCH APPROXIMATE FIRST :k ROWS ONLY
    """, {"vec": vec_str, "k": top_k})

    results = cursor.fetchall()
    latency = time.time() - start_time

    cursor.close()

    # Extrai apenas os IDs
    retrieved_ids = [row[0] for row in results]

    return retrieved_ids, latency


def worker_thread(thread_id: int, query_indices: List[int], result_list: List[QueryResult],
                  query_vectors: np.ndarray, top_k: int):
    """
    Thread worker que processa um subconjunto específico de queries.

    Args:
        thread_id: ID da thread
        query_indices: lista de índices de queries para esta thread processar
        result_list: lista compartilhada para armazenar resultados
        query_vectors: vetores de consulta (array completo)
        top_k: número de vizinhos a retornar
    """
    # Cada thread tem sua própria conexão
    connection = connect_database()

    try:
        # Processa apenas as queries atribuídas a esta thread
        for query_id in query_indices:
            # Executa busca vetorial
            query_vector = query_vectors[query_id]
            retrieved_ids, latency = vector_search(connection, query_vector, top_k)

            # Armazena resultado
            result = QueryResult(
                query_id=query_id,
                latency=latency,
                retrieved_ids=retrieved_ids
            )
            result_list.append(result)

            # Pausa de 5ms entre queries para não sobrecarregar o banco
            time.sleep(0.150)

    finally:
        connection.close()


def distribute_queries(num_queries: int, num_threads: int) -> List[List[int]]:
    """
    Distribui as queries entre as threads de forma balanceada.

    Args:
        num_queries: número total de queries
        num_threads: número de threads

    Returns:
        List[List[int]]: lista de listas, onde cada sublista contém os índices
                         das queries para uma thread específica
    """
    queries_per_thread = num_queries // num_threads
    remainder = num_queries % num_threads

    distributions = []
    current_idx = 0

    for i in range(num_threads):
        # Threads iniciais recebem uma query extra se houver resto
        size = queries_per_thread + (1 if i < remainder else 0)
        distributions.append(list(range(current_idx, current_idx + size)))
        current_idx += size

    return distributions


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calcula percentil de uma lista de valores"""
    sorted_values = sorted(values)
    index = int(len(sorted_values) * percentile)
    return sorted_values[min(index, len(sorted_values) - 1)]


def run_benchmark(query_vectors: np.ndarray, num_threads: int = 4, top_k: int = 100) -> Dict:
    """
    Executa benchmark de busca vetorial com múltiplas threads.
    Cada thread processa um subconjunto exclusivo de queries.

    Args:
        query_vectors: vetores de consulta
        num_threads: número de threads paralelas
        top_k: número de vizinhos a retornar

    Returns:
        Dict: dicionário com estatísticas do benchmark
    """
    num_queries = len(query_vectors)
    print(f"\n{'='*60}")
    print(f"Iniciando Benchmark Openai 1536D 1M")
    print(f"{'='*60}")
    print(f"Queries: {num_queries}")
    print(f"Threads: {num_threads}")
    print(f"Top-K: {top_k}")

    # Distribui queries entre threads
    query_distributions = distribute_queries(num_queries, num_threads)

    print(f"\n--- DISTRIBUIÇÃO DE QUERIES ---")
    for i, indices in enumerate(query_distributions):
        print(f"Thread {i}: {len(indices)} queries (índices {indices[0]} a {indices[-1]})")
    print(f"{'='*60}\n")

    # Lista compartilhada para resultados (thread-safe com append)
    results = []

    # Cria e inicia threads
    threads = []
    start_time = time.time()

    for i in range(num_threads):
        thread = threading.Thread(
            target=worker_thread,
            args=(i, query_distributions[i], results, query_vectors, top_k)
        )
        thread.start()
        threads.append(thread)

    # Aguarda threads finalizarem
    for thread in threads:
        thread.join()

    total_time = time.time() - start_time

    # Calcula estatísticas
    latencies = [r.latency for r in results]

    stats = {
        "total_queries": num_queries,
        "total_time": total_time,
        "queries_per_second": num_queries / total_time,
        "avg_latency": np.mean(latencies),
        "min_latency": np.min(latencies),
        "max_latency": np.max(latencies),
        "p90_latency": calculate_percentile(latencies, 0.90),
        "p95_latency": calculate_percentile(latencies, 0.95),
        "p99_latency": calculate_percentile(latencies, 0.99)
    }

    return stats


def print_statistics(stats: Dict):
    """
    Imprime estatísticas do benchmark formatadas.

    Args:
        stats: dicionário com estatísticas
    """
    print(f"\n{'='*60}")
    print(f"RESULTADOS DO BENCHMARK")
    print(f"{'='*60}")
    print(f"Total de Queries:        {stats['total_queries']}")
    print(f"Tempo Total:             {stats['total_time']:.2f}s")
    print(f"\n--- THROUGHPUT ---")
    print(f"Queries por Segundo:     {stats['queries_per_second']:.2f} QPS")
    print(f"\n--- LATÊNCIA ---")
    print(f"Latência Média:          {stats['avg_latency']*1000:.2f}ms")
    print(f"Latência Mínima:         {stats['min_latency']*1000:.2f}ms")
    print(f"Latência Máxima:         {stats['max_latency']*1000:.2f}ms")
    print(f"Latência P90:            {stats['p90_latency']*1000:.2f}ms")
    print(f"Latência P95:            {stats['p95_latency']*1000:.2f}ms")
    print(f"Latência P99:            {stats['p99_latency']*1000:.2f}ms")
    print(f"{'='*60}\n")


def main():
    """Função principal que executa o benchmark"""

    # Parser de argumentos
    parser = argparse.ArgumentParser(description='Vector Search Benchmark - Oracle AI')
    parser.add_argument('--threads', type=int, default=4,
                       help='Número de threads paralelas (default: 4)')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Número de vizinhos mais próximos (default: 100)')
    parser.add_argument('--query-file', type=str, default='query_vectors_10000.csv',
                       help='Arquivo de queries CSV (default: query_vectors_10000.csv)')

    args = parser.parse_args()

    try:
        # 1. Carrega queries
        print(f"Carregando queries de '{args.query_file}'...")
        query_vectors = read_query_vectors_csv(args.query_file)
        print(f"✓ {len(query_vectors)} queries carregadas (dim={query_vectors.shape[1]})")

        # 2. Executa benchmark
        stats = run_benchmark(
            query_vectors=query_vectors,
            num_threads=args.threads,
            top_k=args.top_k
        )

        # 3. Imprime estatísticas
        print_statistics(stats)
    except FileNotFoundError as e:
        print(f"✗ Erro: Arquivo não encontrado - {e}")
    except Exception as e:
        print(f"✗ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()