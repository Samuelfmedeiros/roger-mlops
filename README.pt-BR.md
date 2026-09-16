<p align="center">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/licen%C3%A7a-MIT-6366f1?style=flat-square" alt="Licença">
  </a>
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/python-3.10%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  </a>
  <img src="https://img.shields.io/badge/depend%C3%AAncias-0-22c55e?style=flat-square" alt="Zero dependências">
  <img src="https://img.shields.io/badge/SO-Windows%2011%20%2B%20WSL2-fuchsia?style=flat-square" alt="Windows + WSL2">
</p>

<h1 align="center">Roger — MLOps local resiliente</h1>

<p align="center"><strong>Mantém corridas longas de treino de GPU sem supervisão honestas em WSL2 e hardware edge</strong> — crítico determinístico, watchdogs que agem e I/O de checkpoint à prova de queda. Zero dependências, agnóstico de modelo.</p>

> 🇧🇷 **Português** · [🌐 English](README.md)

Treino longo de GPU sem supervisão numa estação Windows + WSL2 falha onde ninguém está olhando: o treinador fica vivo enquanto o log para, a ponte NVML morre no sleep/hibernação, o checkpoint retoma por cima de um save truncado, e o loop de agente herda de manhã a nota perfeita de ontem. Nada disso gera erro às 03:00 — a corrida só piora em silêncio.

Roger é a camada de guardas que transforma cada um desses em **evento tratado**:

- um **crítico determinístico** que pontua a corrida contra as curvas que você declarou (LR vs schedule, norma de gradiente, velocidade de loss, tokens/s, VRAM, estagnação do log) e arquiva cada veredito com sua evidência num ledger append-only;
- **watchdogs que agem** — matam o treinador travado e relançam do último checkpoint bom, sob orçamento de relançamento com breaker de loop de queda;
- uma **cura da ponte GPU-PV** que executa o reinício cirúrgico do WSL2 que o guest nunca pode pedir para si mesmo;
- **I/O de checkpoint à prova de queda** feito para 9p/drvfs: um save ou existe completo ou não existe — o estado meio-escrito que corrompe o resume sai do campo do possível;
- **higiene de orquestrador** que mantém o loop de agente honesto: saída de ferramenta não confiável quarentenada como dado, fingerprint de estado por campanha, sentinelas de regressão e de rodada inconclusiva.

Zero dependências. Agnóstico de modelo. Cada guarda carrega um teste unitário escrito contra o incidente que a gerou. Se você treina no seu próprio hardware, esta é a camada de confiabilidade que seu escalador deveria ter trazido desde o primeiro dia.

## Os modos de falha, e a guarda que mata cada um

| Sintoma (silencioso sem guarda) | Guarda | Módulo |
|---|---|---|
| Ponte NVML `found a PCI device but no GPUs found` após sleep/hibernação; guest saudável pra sempre | Cura do lado host: sonda via ponte, `wsl --shutdown` cirúrgico + relançamento, arbitragem de TFLOPS reais antes de confiar no canal | `tatu/gpu_cure.py`, `scripts/cuda_tflops_bench.py` |
| Processo de treino vivo porém travado: CPU 100%, log congelado, GPU 0 | Mate-zumbi com orçamento de relançamento; detecção de regressão de passo; breaker de loop de queda | `tatu/night_watch.py` |
| Host reporta "2 GB usados pelo WSL" — OOM mascarado em memória compartilhada | Sonda da verdade do host: enumera processos da VM, soma WS, atribui VRAM por processo | `scripts/host_vram_probe.ps1` |
| Checkpoint truncado em voo no 9p/drvfs e o resume passa por cima do lixo | `.part` + fsync + verificação por leitura + `os.replace`; quarentena fora do glob dos resolvers | `tatu/safe_io.py`, `scripts/quarantine.py` |
| Crítico via HTTP/JSON: frágil, muralha de auth, timeout | IPC por arquivo em ext4 com flock, modo one-shot, veredictos fail-open, ledger append-only | `tatu/roger_tatu.py`, `tatu/roger_client.py` |
| Um loop de agente herda o 100/100 de ontem porque o arquivo de estado sobreviveu à campanha | Fingerprint de campanha (git HEAD) + reconciliação de estado + detecção de regressão | `tatu/hygiene.py` |
| A saída de ferramenta do repo auditado injeta instruções no seu agente | Quarentena de conteúdo não confiável (cercas, caracteres de controle, truncamento) | `tatu/hygiene.py` |

## Arquitetura

```
 treinador (qualquer framework, qualquer máquina)
   |  checkpoints atômicos + verificados ....... tatu/safe_io.py
   |  pergunta "devo continuar?" (não bloqueante, fail-open 90s)
   v
 daemon crítico Roger  <── IPC por arquivo, flock em ext4, nunca 9p/HTTP
   |  score determinístico 0-100 + veredito {continue|watch|investigate|escalate}
   |  ledger.jsonl (evidência append-only)
   v
 night_watch (guest) — zumbi/regressão de passo/loop de queda -> mata + relança
 gpu_cure   (host)  — cura da ponte NVML/dxgkrnl, reinício cirúrgico do WSL
 keepalive (guest) — loop de relançamento resume-safe (systemd --user)
```

O crítico **nunca bloqueia o treino** — toda dependência falha aberta com alerta. A opinião do Roger é consultiva por contrato: ele gateia a *campanha*, não a *época*.

## Instalação

Sem dependências. Python 3.10+ de cada lado da ponte.

```bash
git clone <este-repo> && cd roger-mlops
cp .env.example ~/.config/roger.env   # edite os paths para o seu layout
export TATU_HOME=$HOME/.tatu TATU_TRAIN_LOG=$HOME/.tatu/train.log
python3 -m unittest discover -s tests -v   # tudo verde antes de confiar
```

As units systemd de usuário estão em `deploy/`:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/roger-tatu.service deploy/keepalive.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now roger-tatu keepalive
```

## Comandos

| Comando | O que faz |
|---|---|
| `python3 tatu/roger_tatu.py --daemon` | Roda o daemon crítico (IPC por arquivo, spool com flock) |
| `python3 tatu/roger_tatu.py --once` | Drena requisições pendentes uma vez (amigável a cron, sem daemon) |
| `python3 tatu/roger_client.py ask --project NOME --log PATH [--dry-run]` | Pede veredito; imprime `VERDICT=`/`SCORE=`; fail-open |
| `python3 tatu/night_watch.py --once` | Um check de zumbi/regressão/loop; mata + relança |
| `python3 tatu/gpu_cure.py --probe-only` | Classifica o estado da ponte GPU sem curar |
| `python3 tatu/hygiene.py` | Self-test de toda guarda de orquestrador |
| `python3 scripts/quarantine.py move DIR CKPT` | Tira um checkpoint suspeito do glob dos resolvers |
| `python3 scripts/cuda_tflops_bench.py` | Mede TFLOPS reais (o árbitro de throughput da GPU) |
| `pwsh scripts/host_vram_probe.ps1` | Verdade do host: working set da VM + VRAM por processo |
| `bash deploy/launch_train.sh` | Launcher de referência resume-safe (flock + pgrep em dobra) |

## Contrato do crítico

Requisições e veredictos são JSON via arquivos `request_*.json` (schema em
`tatu/roger_tatu.py`); os checks determinísticos cobrem as curvas que um humano
olharia às 03:00 — LR vs o schedule cosseno declarado, explosão de gradiente,
velocidade de loss, deriva de campos (ex.: `|A_log|`), tok/s, VRAM,
estagnação do log, presença do processo — cada um com peso e gate. Se você
plugar um crítico no seu próprio loop, ele deve imprimir `SCORE=<n>` /
`GAPS=<a|b>` / `DETAIL=<...>`; `hygiene.parse_critic_output()` se recusa a
notar 0 um crítico que quebrou o contrato, em vez de gerar uma rodada falsa
de defeito.

## Por que arquivos, não sockets

Num laptop de armazenamento híbrido sob carga de treino, as pontes Windows↔WSL mentem: leituras de metadados de diretório dão erro de I/O, a saída do `wsl.exe` chega vazia, portas em localhost espelhado pertencem ao `wslrelay` mesmo com o serviço morto, e o `df` via 9p reporta números de cache. Uma requisição escrita em ext4 com flock e fsync ou existe completa ou não existe — não há terceiro estado. Essa propriedade é o projeto inteiro.

## Testes

```bash
python3 -m unittest discover -s tests -v
```

Cobrem: pesos do veredito e gate, fail-open de requisição malformada, recusa
de daemon duplicado, tentativas de burlar a quarentena, arquivamento de nota
herdada, gap fluctuante (regressão), rodada de ambiente `0 passed`,
verificação de checkpoint contra truncamento e o parse do marcador de
conclusão do launcher.

## Licença

MIT — ver [LICENSE](LICENSE).
