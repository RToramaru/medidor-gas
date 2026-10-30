# Leitor OCR de hidrômetros

Aplicativo desktop para ler hidrômetros a partir de imagens, vídeos e câmera. A imagem e a leitura aparecem na mesma tela; o processamento é feito no próprio computador.

## Recursos

- Detecta e reconhece os dígitos do visor.
- Salva leituras em CSV e grava vídeos em MP4.
- Reutiliza a calibração da fonte quando ela já existe.
- Mostra somente a imagem original: nenhum desenho, caixa ou rótulo é sobreposto.
- Exibe e permite salvar a leitura assim que todos os dígitos forem reconhecidos.
- Permite selecionar e reconectar câmeras e ajustar regiões com coordenadas.
- Processa imagens, vídeos e câmera em segundo plano para manter a interface responsiva.

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
3. Aguarde a leitura inicial. O valor aparece assim que todos os dígitos forem reconhecidos.
4. Se necessário, clique em **Calibrar novamente** ou em **Ajustar regiões**. O ajuste é feito com coordenadas numéricas, sem desenhar na imagem.
5. Em vídeo e câmera, use **Pausar** e **Gravar vídeo**. **Salvar leitura** fica disponível quando a leitura está completa.

Imagens são analisadas uma vez. A leitura é uma estimativa e deve ser conferida antes de ser usada como registro oficial. Caso uma câmera ou vídeo continue aberto após um erro de leitura, tente calibrar novamente.

## Arquivos gerados

Os resultados ficam na pasta de dados do usuário:

- Windows: `%LOCALAPPDATA%\LeitorGas\resultados`
- macOS: `~/Library/Application Support/LeitorGas/resultados`
- Linux: `$XDG_DATA_HOME/leitor-gas/resultados` ou `~/.local/share/leitor-gas/resultados`

Essa pasta contém `leituras.csv`, `calibracoes/` e os vídeos gravados. Na primeira execução, calibrações e CSV existentes em `resultados/` na pasta do projeto são copiados para o novo local sem remover os originais. Mantenha os modelos incluídos junto ao programa.

## Criar um aplicativo distribuível

Faça o build no próprio sistema operacional em que o aplicativo será executado:

```shell
python -m pip install -r requirements-build.txt
python build.py
```

O pacote fica em `dist/LeitorHidrometros/`. O PyInstaller não gera executáveis de outros sistemas operacionais; faça um build nativo separado para Windows, macOS e Linux.

## Organização

```text
leitor-ocr-gas/
├── main.py
├── leitor_gas/
│   ├── application.py
│   ├── cameras.py
│   ├── persistence.py
│   ├── settings.py
│   └── vision.py
├── modelos/
├── requirements.txt
├── requirements-build.txt
├── build.py
└── tests/
```

- `main.py` inicia o aplicativo.
- `leitor_gas/application.py` contém a interface e o ciclo de vida de imagens, vídeos e câmera.
- `leitor_gas/cameras.py` e `persistence.py` cuidam da busca de câmeras e do armazenamento.
- `leitor_gas/vision.py` contém a detecção e a classificação dos dígitos.
- `leitor_gas/settings.py` centraliza caminhos, parâmetros e cores.
- `build.py` cria o pacote distribuível para o sistema atual.

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
