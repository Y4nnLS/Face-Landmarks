from termcolor import colored

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

main_menu()