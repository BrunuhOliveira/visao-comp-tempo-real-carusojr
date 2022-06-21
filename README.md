# Deep Learning e Visão Computacional para Detecção de Imagens
Detecção de imagem em tempo real utilizando Deep Learning em Florianópolis, utilizando arquitetura MobileNet para redes neurais convolucionais para detecção de magens em tempo real.

Para detecção de imagens, foi utilizado a estratégia de aprendizado supervisionado, onde utilizei os datasets (ImageNet) do Google para obtenção dos dados de entradas e definição das classes de saída.

Sendo assim foi gerado a rede neural convolucional, com os devidos pesos calculados e utilizado transferlearning para exportação do modelo e aplicação.

Foi utilizado o pacote OpenCv para execução do modelo já treinado e utilizado junto com o framework de deeplearning Caffe.

## Arquivos presentes:

MobileNetSSD_deploy.caffemodel - Arquivo onde tem os devidos pesos já calculados para a rede neural e classificação das imagens.

MobileNetSSD_deploy.prototxt.txt - Arquivo contando a especificação do modelo, como será executado e as ordens de execução de cada camada da rede neural (Convolucional, MaxPooling, Relu...)

deteccao-tempo-real-caruso.py - Arquivo que irá executar e orquestrar a junção do modelo treinado juntamente com o OpenCv, para execução do vídeo.

## Como executar:

Para executar o código é necessário que seja realizado o download dos vídeos obtidos do Youtube, já baixados e disponíveis no link abaixo:

https://drive.google.com/drive/folders/1vfjViZP-XNFhMnLCpUTlqSeuFumwTi30

Após baixado, esses vídeos tem que serem copiados para a mesma pasta do projeto, para que seja executado automaticamente pelo programa, conforme imagem abaixo:

![image](https://user-images.githubusercontent.com/24361738/174684668-9d2d296c-b75c-47c8-ae3a-6a63c97fbff2.png)

Talvez seja necessário instalar os pacotes python abaixo para execução do programa.

**pip install opencv-python**

**pip install imutils**

**pip install argparse**

Estando o terminal no diretório do arquivo, basta executar o comando:

**python deteccao-tempo-real-caruso.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel**

## Referências e Fontes:

4K | Dirigindo em Florianópolis - da Ponte a Canasvieiras - Driving 4K Brazil - ep. 70
https://www.youtube.com/watch?v=4zbOtGcJW40

FLORIANÓPOLIS • DRIVING • Dirigindo Brasil【4K 60fps】Floripa • Centro
https://www.youtube.com/watch?v=0G6DewSJpfU

https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html

## Resultados Obtidos

![image](https://user-images.githubusercontent.com/24361738/174685601-e990a121-915d-413b-8708-9c456f895c9e.png)

![image](https://user-images.githubusercontent.com/24361738/174685660-9d943766-5cc9-45b6-a573-393bf623b901.png)

![image](https://user-images.githubusercontent.com/24361738/174685780-df31ef19-c881-4f85-a491-74e06194027e.png)

![image](https://user-images.githubusercontent.com/24361738/174685848-b6964c2c-bcaf-4bd1-8a2f-b288c09062e1.png)
