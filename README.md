# PINNs - Redes Neurais Informadas pela Física

## Sobre o Projeto
O **PINNs** (“Physics-Informed Neural Networks”) é um repositório focado no estudo e implementação de Redes Neurais Informadas pela Física. Essas redes são projetadas para resolver equações diferenciais parciais (PDEs) incorporando informações físicas diretamente na função de perda, permitindo aprender soluções precisas com menos dados.

Este projeto utiliza a biblioteca **DeepXDE**, uma ferramenta poderosa para resolver equações diferenciais usando redes neurais profundas.

## Recursos
- Implementação de PINNs para resolver PDEs.
- Casos de estudo como equação do calor e pulsos gaussianos.
- Utiliza **DeepXDE**, **TensorFlow** e **PyTorch** para treinamento de redes neurais.
- Suporte para visualização e animação dos resultados.

## Estrutura do Repositório
```
PINNs/
├── gaussian_pulse/         # Implementações para pulsos gaussianos
├── heat_equation/          # Estudos sobre a equação do calor
├── network/                # Modelos de redes neurais
├── .gitignore              # Arquivos ignorados pelo Git
├── README.md               # Documentação do projeto
├── animation.py            # Gera animações dos resultados
├── cleanup.sh              # Script para limpeza de arquivos gerados
```

## Instalação
1. Clone o repositório:
   ```bash
   git clone https://github.com/josecamata/pinns.git
   cd pinns
   ```
2. Instale as dependências necessárias:
   ```bash
   pip install -r requirements.txt
   ```
   *Caso o arquivo `requirements.txt` não esteja presente, instale bibliotecas como DeepXDE, TensorFlow, PyTorch, Matplotlib e NumPy manualmente.*



## Como Utilizar
Para rodar a implementação de uma PINN, utilize um dos notebooks disponíveis nos diretórios correspondentes, por exemplo:
```bash
cd heat_equation
python main.py
```

## Limpeza de Arquivos
Se necessário, utilize o script `cleanup.sh` para remover arquivos temporários:
```bash
chmod +x cleanup.sh
./cleanup.sh
```

## Contribuição
Contribuições são bem-vindas! Para contribuir:
1. Faça um fork do repositório.
2. Crie uma branch para suas alterações:
   ```bash
   git checkout -b minha-feature
   ```
3. Faça commit das suas alterações:
   ```bash
   git commit -m "Adicionando nova feature"
   ```
4. Envie para seu repositório remoto:
   ```bash
   git push origin minha-feature
   ```
5. Abra um Pull Request detalhando suas alterações.


## Agradecimento
Este projeto recebeu financiamento da FAPEMIG (APQ-01123-21).

