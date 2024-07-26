# Data Science - DCA0131 🗄️

Files, developed throughout the 2024.1 semester of the Data Science discipline taught at the Federal University of Rio Grande do Norte by the Department of Computer Engineering and Automation (DCA). 📚

## Development 🚀

### Prerequisites ⚙️

- Python 3.12.2 🐍
- pip 24.0 📦

### Requirements 📋

```bash
pip install -r requirements.txt
```

> If necessary, use [virtualenv](./docs/using-virtualenv.md). 🔄

```bash
# Navegue para o diretório do seu projeto
cd path

# Desative o ambiente virtual, se estiver ativado
deactivate  # Se o venv estiver ativado, isso o desativa

# Verifique o caminho do Python atual (deve ser o global, não o do venv)
which python

# Verifique o caminho do pip atual (deve ser o global, não o do venv)
which pip  # Certifique-se de que o pip está no mesmo caminho do Python

# Remova o ambiente virtual antigo, se existir
rm -rf venv  # Apaga o diretório 'venv' e seu conteúdo

# Crie um novo ambiente virtual
python3 -m venv venv  # Cria um novo ambiente virtual chamado 'venv'

# Ative o novo ambiente virtual
source venv/bin/activate  # Ativa o ambiente virtual recém-criado

# Verifique o caminho do Python novamente (deve ser o do venv agora)
which python

# Verifique o caminho do pip novamente (deve ser o do venv agora)
which pip  # Certifique-se de que pip está no mesmo caminho do Python no venv

# Instale as dependências listadas em requirements.txt
pip install -r requirements.txt  # Instala os pacotes necessários listados em requirements.txt

```

### Execute ▶️

```bash
python3 main.py
```