# libra
Este trabalho esta estruturado em 2 linguagens diferentes. C++ e Java.

Tudo que estiver relacionado aos descritores, estara definido dentro de .cpp sendo esta contendo o cdigo para gerar as características de todo o dataset, quanto pegar do vídeo. Estas duas funcionalidades devem ser descomentadas quando utilizadas. O vídeo deverá escrever o resultado do descritor no mesmo arquivo que java executará a leitura

Já as operaçes relacionadas à clasificação estarão contidas nos arquivos .java que faram a leitura das caractersticas geradas pela execução do .cpp

Para rodar .jar deve-se entrar na pasta ClassificadorLibras e digitar no terminal "java -jar classificadorLibras_HU.jar predict data/testeConsumidor_HU.arff" que o treinamento e a leitura se iniciará. Para rodar o binário do descritor vá em bin/Debug e execute "./libra"
