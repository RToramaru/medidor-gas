# Leitor OCR de hidrômetros

Aplicativo desktop para ler hidrômetros a partir de imagens, vídeos e câmera. A imagem e a leitura aparecem na mesma tela; o processamento é feito no próprio computador.

## Recursos

- Abrir imagem, vídeo ou câmera local.
- Calibrar automaticamente e refazer a calibração quando necessário.
- Pausar fontes de vídeo e câmera.
- Salvar leituras em CSV e gravar vídeos em MP4.

## Requisitos

- Python 3.11 recomendado.
- Ambiente gráfico com suporte a Tkinter.
- Dependências Python do arquivo [`requirements.txt`](requirements.txt).
- Arquivos de modelos incluídos na pasta [`modelos`](modelos).

O código usa caminhos relativos e não depende de comandos exclusivos de um sistema operacional. Pode ser executado em Windows, macOS e Linux, desde que a versão do Python e as dependências estejam disponíveis para o sistema e processador. No Linux, instale também o pacote Tkinter da sua distribuição (por exemplo, `python3-tk` no Debian/Ubuntu). A câmera pode exigir permissão do sistema.

Não é necessária uma placa de vídeo; a leitura também pode ser processada pela CPU.

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

1. Aguarde o aplicativo ficar pronto.
2. Selecione **Abrir vídeo**, **Abrir imagem** ou **Câmera local**.
3. Aguarde a leitura inicial. Use **Calibrar** se os dígitos não forem identificados corretamente.
4. Em vídeos e câmera, use **Pausar** e **Gravar** conforme necessário. **Salvar leitura** acrescenta o valor atual ao histórico.

Imagens são analisadas uma vez. A leitura é uma estimativa e deve ser conferida antes de ser usada como registro oficial.

## Arquivos gerados

O aplicativo cria `resultados/` na pasta do projeto:

- `leituras.csv`: histórico de leituras salvas.
- `calibracoes/`: calibrações reutilizadas ao abrir a mesma fonte novamente.
- `video_*.mp4`: gravações feitas pelo aplicativo.

Mantenha a pasta `modelos/` e seu conteúdo junto ao programa. O repositório precisa incluir também a pasta `stage1_meter_window_digits_openvino_model` completa.

## Problemas comuns

- **O aplicativo não inicia:** confira a instalação das dependências e se Tkinter está disponível no Python usado.
- **Leitura incorreta:** verifique se o visor está nítido e clique em **Calibrar**.
- **Câmera indisponível:** confira a conexão, as permissões do sistema e se outro aplicativo está usando a câmera.
- **Leitura lenta:** o processamento pode usar CPU e ficar mais lento em vídeos de alta resolução.
