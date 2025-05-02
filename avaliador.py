import os
import sys
import subprocess
import time

def compilar_cpp(caminho_cpp, nome_exec):
    try:
        subprocess.run(["g++", caminho_cpp, "-o", nome_exec], check=True)
        return True
    except subprocess.CalledProcessError:
        print(f"Erro ao compilar {caminho_cpp}")
        return False

def ler_arquivo_completo(caminho):
    with open(caminho, "r") as f:
        return f.read()

def testar_programa(executavel, entrada, saida_esperada):
    inicio = time.perf_counter()
    try:
        resultado = subprocess.run(
            [f"./{executavel}"],
            input=entrada.encode(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10  # segundos
        )
        fim = time.perf_counter()
        tempo_total = fim - inicio

        saida_obtida = resultado.stdout.decode()

        correto = saida_obtida.strip() == saida_esperada.strip()
        return correto, tempo_total, saida_obtida

    except subprocess.TimeoutExpired:
        print("Tempo de execução excedido.")
        return False, None, ""

def main():
    if len(sys.argv) != 2:
        print("Uso: python avaliador.py nome_da_pasta")
        return

    pasta = sys.argv[1]
    nome_base = os.path.basename(pasta.rstrip("/"))

    cpp_path = os.path.join(pasta, f"{nome_base}.cpp")
    input_path = os.path.join(pasta, "input.txt")
    output_path = os.path.join(pasta, "output.txt")

    if not all(os.path.exists(p) for p in [cpp_path, input_path, output_path]):
        print("Arquivos necessários não encontrados:")
        print(f"Esperado: {nome_base}.cpp, input.txt, output.txt dentro da pasta {pasta}")
        return

    exec_name = f"{nome_base}_exec"

    if not compilar_cpp(cpp_path, exec_name):
        return

    entrada = ler_arquivo_completo(input_path)
    saida_esperada = ler_arquivo_completo(output_path)

    correto, tempo, saida_obtida = testar_programa(exec_name, entrada, saida_esperada)

    print(f"\n--- Resultado para {nome_base} ---")
    print(f"Correto? {'Sim' if correto else 'Não'}")
    print(f"Tempo de execução: {tempo:.4f} segundos")
    if not correto:
        print("\n--- Saída obtida ---")
        print(saida_obtida)

if __name__ == "__main__":
    main()
