# Reproduzindo Attetion is All You Need

Objetivo deste projeto é reproduzir o artigo que propôs a arquitetura dos Transformers. Essa rede neural é 
especialmente boa em NLP. O artigo usa o problema de tradução para demonstrar a eficácia da sua rede. Traduzindo de 
inglês para alemão e de inglês para francês.

## Dataset

De início, foi obitido o dataset de tradução de inglês para alemão. O dataset foi colocado na pasta 
**tff/English-german**.

## Implementações

Foram tentadas 3 implementações. A primeira foi usando tensorflow, mas por terem indícios de ser mais lenta ou/e mais
custosa, não foi usada mais. Essa primeira versão está no caminho **tff/tensorflow**. A segunda implementação usa pytorch, 
mas após alguns problemas no tokenizer, foi preferível usar uma implementação que baseava o tokenizer no próprio 
data-set. Essa implementação está no caminho **tff/pytorch1**. Por fim, a última implementação, também usando pytorch está 
no caminho **tff/pytorch2**.

## Docker

Para rodar o container docker eu sempre usava o comando abaixo. Desse modo habilitava as GPUS dentro do container e
compartilhava a pasta tff para dentro do container.

```
docker run --gpus all -it -v /tff:/tff --rm $IMAGENAME bash
```

### Imagens

Foram usadas 4 imagens para executar os testes. Para o tensorflow foi usada inicialmente uma imagem padrão do 
tensorflow, a **tensorflow/tensorflow:devel-gpu**. Depois de alguns problemas com uma versão antiga do CUDA, foi usado uma
imagem própria que eu havia usado em outros projetos. Ela se encontra em **dockerfile-tff/Dockerfile**. Para ela
funcionar por completo, é necessário rodar o bash script dentro do container **tff/tff_install_federated_py310**. Para o 
pytorch, usei a imagem padrão fornecida pela biblioteca, a **pytorch/pytorch:latest**. Após alguns problemas com a versão do
CUDA, encontrei uma fornecida pela nvidia com uma versão mais recente, a **nvcr.io/nvidia/pytorch:23.07-py3**.