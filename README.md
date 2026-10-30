# Leitor OCR de hidrômetros

Aplicativo desktop para ler hidrômetros a partir de imagens, vídeos e câmera. A imagem e a leitura aparecem na mesma tela; o processamento é feito no próprio computador.

## Recursos

- Detecta e reconhece os dígitos do visor.
- Salva leituras em CSV e grava vídeos em MP4.
- Reutiliza a calibração da fonte quando ela já existe.

O programa usa detecção YOLO para encontrar os dígitos e classificadores TFLite para ler os valores. A aceleração por GPU é opcional; também é possível executar em CPU.

## Requisitos

- Python 3.11 recomendado.
- Ambiente gráfico com suporte a Tkinter.
- Dependências Python do arquivo [`requirements.txt`](requirements.txt).
- Arquivos de modelos incluídos na pasta [`modelos`](modelos).

Pode ser executado em Windows, macOS e Linux, desde que Python, Tkinter e as dependências estejam disponíveis para o sistema e o processador. No Linux, pode ser necessário instalar o Tkinter pelo gerenciador de pacotes (por exemplo, `python3-tk` no Debian/Ubuntu). A câmera pode exigir permissão do sistema.

## Clonar

Substitua `<URL-do-repositorio>` pela URL do repositório:

```shell
git clone <URL-do-repositorio>
cd leitor-ocr-gas
```

## Instalar

Crie um ambiente virtual na pasta do projeto.

**Windows (PowerShell):**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS e Linux:**

```shell
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Executar

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\python.exe main.py
```

**macOS e Linux**, com o ambiente virtual ativado:

```shell
python main.py
```

## Usar

1. Aguarde o carregamento inicial.
2. Selecione uma imagem, vídeo ou câmera no menu lateral.
3. Aguarde a leitura inicial; se necessário, clique em **Calibrar novamente**.
4. Em vídeo e câmera, use **Pausar** e **Gravar vídeo**. **Salvar leitura** acrescenta o valor atual ao histórico.

Imagens são analisadas uma vez. A leitura é uma estimativa e deve ser conferida antes de ser usada como registro oficial. Caso uma câmera ou vídeo continue aberto após um erro de leitura, tente calibrar novamente.

## Arquivos gerados

O aplicativo cria `resultados/` na pasta do projeto:

- `leituras.csv`: histórico de leituras salvas.
- `calibracoes/`: calibrações reutilizadas ao abrir a mesma fonte novamente.
- `video_*.mp4`: gravações feitas pelo aplicativo.

Mantenha a pasta `modelos/` e seu conteúdo junto ao programa. Os arquivos de calibração existentes são mantidos em `resultados/calibracoes/`.

## Organização

```text
leitor-ocr-gas/
├── main.py
├── leitor_gas/
│   ├── application.py
│   ├── settings.py
│   └── vision.py
├── modelos/
├── requirements.txt
├── resultados/  # criado automaticamente; ignorado pelo Git
└── tests/
```

- `main.py` inicia o aplicativo.
- `leitor_gas/application.py` contém a interface e o ciclo de vida de imagens, vídeos e câmera.
- `leitor_gas/vision.py` contém a detecção e a classificação dos dígitos.
- `leitor_gas/settings.py` centraliza caminhos, limiares e cores.

## Testes

Execute os testes automatizados com:

```shell
python -m unittest discover -s tests -v
```

## Problemas comuns

- **O aplicativo não inicia:** confira a instalação das dependências e se Tkinter está disponível no Python usado.
- **Erro ao carregar os modelos:** verifique se os arquivos de `modelos/` estão presentes.
- **Leitura incorreta:** verifique se o visor está nítido e clique em **Calibrar**.
- **Câmera indisponível:** confira a conexão, as permissões do sistema e se outro aplicativo está usando a câmera.
- **Leitura lenta:** o processamento pode usar CPU e ficar mais lento em vídeos de alta resolução.
