import time
import sys
from termcolor import colored
from predict import prediction
import os

def loading_screen():
    print("Loading", end="")
    for _ in range(10):
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(0.1)
    print("\n")

def main_menu():
    print("+-----------------------------------------+")
    print("|", end=' ')
    print(colored("Selecione a predição que deseja testar:", "cyan"),end=' ')
    print("|")
    print("""| 1. LBPH                                 |
| 2. Fisherface                           |
| 3. Eigenface                            |
| 4. Sair                                 |
+-----------------------------------------+""")

def main():
    while True:
        loading_screen()
        main_menu()
        choice = input(colored("Digite o número da sua escolha: ", "yellow"))

        if choice == '1':
            print(colored("\nExecutando predição LBPH...\n", "green"))
            prediction.predict('1')
        elif choice == '2':
            print(colored("\nExecutando predição Fisherface...\n", "green"))
            prediction.predict('2')
        elif choice == '3':
            print(colored("\nExecutando predição Eigenface...\n", "green"))
            prediction.predict('3')
        elif choice == '4':
            print(colored("\nSaindo...\n", "red"))
            break
        else:
            print(colored("\nOpção inválida! Tente novamente.\n", "red"))
            time.sleep(1)
            os.system('cls')

if __name__ == '__main__':
    main()
