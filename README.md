# Leitor OCR de hidrômetros

Aplicação desktop para acompanhar a leitura de hidrômetros a partir de uma imagem, de um arquivo de vídeo ou de uma câmera local. A interface mostra a imagem analisada, a leitura reconhecida e o estado da fonte.

Todo o processamento é feito localmente: o programa não depende de API ou de conexão com a internet durante a leitura. A internet é necessária para clonar o repositório e instalar as dependências.

## Funcionalidades

- Abre imagens, vídeos e a câmera local (índice `0`).
- Detecta a posição dos dígitos durante a calibração e apresenta a leitura na interface.
- Reutiliza calibrações salvas para a mesma fonte e resolução.
- Permite pausar e retomar fontes de vídeo e câmera.
- Salva leituras em CSV e grava vídeo em MP4.
- Tenta usar aceleração disponível e recorre a CPU quando necessário.

## Como a leitura funciona

1. Ao abrir uma fonte sem calibração salva, o detector YOLO localiza os dígitos e define as regiões de interesse (ROIs). A calibração também pode ser refeita pelo botão **Calibrar**.
2. Os modelos TFLite classificam os recortes das regiões detectadas. O classificador de 11 classes é usado nos dígitos inteiros; o de 100 classes é usado no último dígito, que representa a casa decimal.
3. Nos frames seguintes, o programa atualiza o último dígito. Quando a casa decimal está entre `0.0` e `0.9`, atualiza também os dígitos anteriores para acompanhar a virada do contador.
4. As regiões e os resultados são exibidos sobre a imagem. A leitura completa aparece quando todos os dígitos inteiros foram reconhecidos.

O detector tenta, nesta ordem, CUDA/NVIDIA, OpenVINO (GPU Intel e depois CPU Intel) e PyTorch/CPU. As opções de aceleração dependem dos componentes e drivers instalados; CPU é a alternativa disponível.

## Requisitos

- Windows 10/11 (interface gráfica e comandos abaixo são para PowerShell).
- Python 3.11 recomendado.
- Dependências listadas em [`requirements.txt`](requirements.txt), incluindo OpenCV, CustomTkinter, Ultralytics/PyTorch, OpenVINO e LiteRT.
- Os arquivos de modelo presentes na pasta [`modelos`](modelos).

Uma GPU compatível é opcional. O aplicativo pode executar a inferência em CPU, embora o desempenho possa ser menor.

## Clonar e instalar

No PowerShell, substitua `<URL-do-repositorio>` pela URL HTTPS ou SSH do repositório:

```powershell
git clone <URL-do-repositorio>
cd leitor-ocr-gas
```

Crie e ative um ambiente virtual e instale as dependências:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Se a política de execução do PowerShell impedir a ativação do ambiente, execute os comandos diretamente pelo interpretador do ambiente virtual:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Executar

Com o ambiente virtual ativado:

```powershell
python main.py
```

Ou sem ativá-lo:

```powershell
.\.venv\Scripts\python.exe main.py
```

Na janela do aplicativo:

1. Aguarde o carregamento dos modelos.
2. Escolha **Abrir vídeo**, **Abrir imagem** ou **Câmera local**.
3. Aguarde a calibração inicial. Se necessário, clique em **Calibrar** para detectar novamente as posições.
4. Use **Pausar** para interromper temporariamente uma fonte de vídeo/câmera, **Gravar** para salvar o vídeo e **Salvar leitura** para registrar o valor atual.
5. Feche o cartão da fonte pelo botão `×`. Fechar a janela encerra o aplicativo.

Imagens são analisadas uma vez. Pausa e gravação estão disponíveis para vídeo e câmera.

## Estrutura do projeto

```text
leitor-ocr-gas/
├── main.py
├── requirements.txt
└── modelos/
    ├── dig-class100-0182-s2_q.tflite
    ├── dig-class11_1910_s2_q.tflite
    ├── stage1_meter_window_digits.pt
    └── stage1_meter_window_digits_openvino_model/
        ├── metadata.yaml
        ├── stage1_meter_window_digits.bin
        └── stage1_meter_window_digits.xml
```

- `main.py`: interface, abertura das fontes, detecção/calibração, classificação e gravação dos resultados.
- `requirements.txt`: dependências Python com versões fixadas.
- `modelos/`: pesos do detector e modelos TFLite usados na classificação. Mantenha os arquivos e a estrutura da pasta ao executar o programa.

## Arquivos gerados

O programa cria a pasta `resultados/` junto a `main.py`:

- `resultados/calibracoes/`: calibrações em JSON, associadas à fonte e à resolução. Elas são reutilizadas ao abrir a mesma fonte novamente.
- `resultados/leituras.csv`: histórico acrescentado pelo botão **Salvar leitura**. As colunas são `data_hora`, `fonte`, `leitura`, `backend` e `rois`; o separador é ponto e vírgula.
- `resultados/video_<id>_<data-hora>.mp4`: gravações iniciadas pelo botão **Gravar**.

## Solução de problemas

- **Erro ao carregar modelos:** confirme que os quatro artefatos estão nos caminhos mostrados na estrutura do projeto e que as dependências foram instaladas no ambiente Python usado para iniciar o programa.
- **Nenhum dígito encontrado:** verifique se a imagem mostra o visor com nitidez e use **Calibrar** para tentar detectar as regiões novamente.
- **Câmera não abre:** confira se a câmera está conectada, se não está sendo usada por outro aplicativo e se o sistema concedeu permissão de acesso.
- **Calibração incompatível:** a calibração requer a mesma resolução da fonte. Clique em **Calibrar** quando a resolução mudar.
- **Desempenho baixo:** sem acelerador compatível, o processamento é feito em CPU; vídeos de maior resolução podem ser mais lentos.

## Observações

- O processamento e os arquivos gerados permanecem na máquina local.
- A leitura é uma estimativa dos modelos de visão computacional; confira o valor reconhecido antes de usá-lo como registro oficial.
- O repositório não informa, neste momento, uma URL de origem para clonagem. Use a URL fornecida pelo mantenedor ou pela plataforma onde o projeto estiver hospedado.
