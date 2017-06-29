package principal;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class Main {
	
	private static final String matrixLabels[][] = {
			{"I", "E", "S", "A"},
			{"W", "QUATRO", "U", "V", "R"},
			{"N", "Q", "M"},
			{"Adulto", "B"},
			{"O", "C", "Pequeno"},
			{"T", "F"},
			{"P", "SETE", "X", "CINCO"},
			{"Aviao", "L", "Y", "Palavra", "G"},
			{"DOIS", "UM"},
			{"Casa", "Pedra", "Verbo", "Junto", "Gasolina", "America"},
			{"Identidade", "Lei"},
			{"D", "NOVE"}		
	};
	
	
	public static void main(String[] args) throws Exception{
		
		// abrindo arquivo de dataset
		DataSource source = new DataSource("data/out321608_final.csv");
		
		// setando classe e excluindo atributos nao uteis
		Instances data = source.getDataSet();
		
		// deletando labels
		data.deleteAttributeAt(data.numAttributes() - 1);
		
		// setando groups como classe
		data.setClassIndex(data.numAttributes() - 1);
		
		// configurando MLP para treinar com selected_data
		MultilayerPerceptron mlpGrupos = new MultilayerPerceptron();
		
		System.out.println("Treinando rede de grupos...");
		mlpGrupos.buildClassifier(data);
		System.out.println("Rede de grupos treinada!");
		
		// lista de datasets dos grupos
		List <Instances> listInstances = new ArrayList<>();
		
		// abrindo datasets dos grupos
		for(int i = 0; i < 12; i++){
			DataSource ds = new DataSource("grupo_" + i + ".csv");
			
			Instances inst = ds.getDataSet();
			
			inst.deleteAttributeAt(inst.numAttributes() - 2);
			inst.setClassIndex(inst.numAttributes() - 1);
			
			listInstances.add(inst);
			
			ds.reset();
		}
		
		// criando as redes para treinar gestos
		List<MultilayerPerceptron> mlpLabel = new ArrayList<>();
		
		// treinando as redes
		for(int i = 0; i < 12; i++){
			MultilayerPerceptron mlp = new MultilayerPerceptron();
			System.out.println("Treinando rede do grupo " + i);
			mlp.buildClassifier(listInstances.get(i));
			System.out.println("Rede do grupo " + i + " treinada!");
			mlpLabel.add(mlp);
		}
		
		DataSource sourceTest;
		Instances dataTest;
		
		BufferedWriter writer;
		File fileGroupsPredict;
		File fileDataset;
		
		/*   DELETANDO ARQUIVOS DE SAIDA PARA NAO DAR ERROS ANTES DE INICIAR AS PREDICOES   */
		File fileTest = new File("data/saida_labels_predict.arff");
		if(fileTest.exists()){
			fileTest.delete();
		}
		/*  -----------------------------------------------------------------------------   */
		
		
		System.out.println("Iniciando predicao...");
		
		BufferedWriter buffLabel = null;
		
		// se for modoConsumidor, entao vai consumir de um arquivo em tempo real
		boolean modoConsumidor = true;
		
		if(args.length >= 1){
			if(args[0].equals("predict")){
				modoConsumidor = true;
			}else{
				if(args[0].equals("eval")){
					modoConsumidor = false;
				}else{
					System.out.println("Argumentos invalidos! Use <predict> ou <eval> \nEncerrando...");
					return;
				}
			}
		}else{
			System.out.println("Necessario especificar tipo de avaliacao! <predict|eval>");
			System.out.println("Encerrando...");
			return;
		}
		
		// usado caso nao seja modoConsumidor
		int count = 0;
		
		// usado para registrar tempo de modificacao do arquivo em tempo real
		long tempoAnterior = 0;
		
		// se nao for modo consumidor entao as saidas sao gravadas num arquivo
		if(!modoConsumidor){
			buffLabel = new BufferedWriter(new FileWriter("data/saida_labels_predict.arff"));
		}
		
		// hashmap para estatisticas
		Map<String, Integer> mapGroups = new HashMap<>();
		Map<String, Integer> mapLabels = new HashMap<>();
		
		String strFile = "";
		if(args.length >= 2){
			strFile = args[1];
		}else{
			System.out.println("Necessario um arquivo arff para testes!\n Encerrando...");
			return;
		}
		
		// roda indefinidamente
		while(true){
			
			// se for modo consumidor entao testa se arquivo de tempo real existe
			if(modoConsumidor){
				fileDataset = new File(strFile);
				if(!fileDataset.exists()){
					System.out.println("Esperando pelo arquivo...");
					Thread.sleep(1000);
					continue;
				}else{
					// resgistra tempo de modificacao
					if(tempoAnterior == 0){
						tempoAnterior = fileDataset.lastModified();
					}else{
						if(tempoAnterior == fileDataset.lastModified()){
							System.out.println("Nenhuma entrada nova no arquivo...");
							Thread.sleep(10000);
							continue;
						}else{
							tempoAnterior = fileDataset.lastModified();
						}
					}
				}
			}
			// abrindo arquivo de dataset para teste
			sourceTest = new DataSource(strFile);
			
			// deletando atributo nao util
			dataTest = sourceTest.getDataSet();
			dataTest.deleteAttributeAt(dataTest.numAttributes() - 1);
			dataTest.setClassIndex(dataTest.numAttributes() - 1);
			
			// se for modo consumidor, pega a ultima - se nao, pega na posicao count
			Instance atual = null;
			if(modoConsumidor){
				if(dataTest.numInstances() >= 1){
					atual = dataTest.lastInstance();
				}else{
					System.out.println("Nenhuma instancia inserida. Aguardando...");
					Thread.sleep(10000);
					continue;
				}
			}else{
				// se tiver pelo menos uma instancia ele faz o teste
				if(dataTest.numInstances() >=1){
					atual = dataTest.instance(count);
				}else{
					System.out.println("Arquivo de instancias para predict vazio. Encerrando...");
					return;
				}
				
			}
			
			// efetua o predict para a instancia 'atual' utilizando a MLP dos grupos
			predict(mlpGrupos, atual, -1);
				
			// arquivo de saida que vai ser consumido depois para determinar o label gesto da instancia
			fileGroupsPredict = new File("data/saida_groups_predict.arff");
			if(fileGroupsPredict.exists()){
				// a flag true determina que vai escrever no fim do arquivo (append)
				writer = new BufferedWriter(new FileWriter("data/saida_groups_predict.arff", true));
				writer.write(atual.toString());
			 	writer.newLine();
				writer.flush();
				writer.close();
			}else{
				// se nao existe entao cria com cabecalho arff
				writer = new BufferedWriter(new FileWriter("data/saida_groups_predict.arff"));
				dataTest.delete();
				writer.write(dataTest.toString());
			 	writer.newLine();
			 	writer.write(atual.toString());
			 	writer.newLine();
			 	writer.flush();
				writer.close();
			}
		
			// limpando e readquirindo dataset de teste
			dataTest.clear();
			dataTest = sourceTest.getDataSet();
			
			// retirando do dataset teste o atributo grupo
			dataTest.deleteAttributeAt(dataTest.numAttributes() - 2);
			 
			// setando o label como classe
			dataTest.setClassIndex(dataTest.numAttributes() - 1);
			
			// pegando instancias com os grupos preditos
			DataSource ds = new DataSource("data/saida_groups_predict.arff");
			Instances inst = ds.getDataSet();
			
			// classe grupos esta na ultima posicao
			inst.setClassIndex(inst.numAttributes() - 1);
			
			// ultima instancia agora vai ser predito seu gesto
			Instance current = null;
			
			if(modoConsumidor){
				current = dataTest.lastInstance();
			}else{
				current = dataTest.instance(count);
			}
			
			Instance instAtual = null;
			if(modoConsumidor){
				instAtual = inst.lastInstance();
			}else{
				instAtual = inst.instance(count);
			}
			
			// se a instancia atual for do grupo zero
			if(instAtual.stringValue(inst.classIndex()).equals("ZERO")){
				
				predict(mlpLabel.get(0), current, 0);
				
				if(modoConsumidor){
					System.out.println("Grupo: ZERO  Gesto: " + current.stringValue(current.classIndex()));
				}else{
					buffLabel.write(current.toString());
					buffLabel.newLine();
					buffLabel.flush();
				}
				
				insertOnHashMap(mapGroups, mapLabels, instAtual, current);
			}
			
			if(instAtual.stringValue(inst.classIndex()).equals("UM")){
				
				predict(mlpLabel.get(1), current, 1);
				
				if(modoConsumidor){
					System.out.println("Grupo: UM  Gesto: " + current.stringValue(current.classIndex()));
				}else{
					buffLabel.write(current.toString());
					buffLabel.newLine();
					buffLabel.flush();
				}
				
				insertOnHashMap(mapGroups, mapLabels, instAtual, current);
				
			}
			
			if(instAtual.stringValue(inst.classIndex()).equals("DOIS")){
				
				predict(mlpLabel.get(2), current, 2);
				
				if(modoConsumidor){
					System.out.println("Grupo: DOIS  Gesto: " + current.stringValue(current.classIndex()));
				}else{
					buffLabel.write(current.toString());
					buffLabel.newLine();
					buffLabel.flush();
				}
				
				insertOnHashMap(mapGroups, mapLabels, instAtual, current);
			}
			
			if(instAtual.stringValue(inst.classIndex()).equals("TRES")){
				
				predict(mlpLabel.get(3), current, 3);
				
				if(modoConsumidor){
					System.out.println("Grupo: TRES  Gesto: " + current.stringValue(current.classIndex()));
				}else{
					buffLabel.write(current.toString());
					buffLabel.newLine();
					buffLabel.flush();
				}
				
				insertOnHashMap(mapGroups, mapLabels, instAtual, current);
			}

			if(instAtual.stringValue(inst.classIndex()).equals("QUATRO")){
				
				predict(mlpLabel.get(4), current, 4);
				
				if(modoConsumidor){
					System.out.println("Grupo: QUATRO  Gesto: " + current.stringValue(current.classIndex()));
				}else{
					buffLabel.write(current.toString());
					buffLabel.newLine();
					buffLabel.flush();
				}
				
				insertOnHashMap(mapGroups, mapLabels, instAtual, current);
			}
						
			if(instAtual.stringValue(inst.classIndex()).equals("CINCO")){
				
				predict(mlpLabel.get(5), current, 5);
				
				if(modoConsumidor){
					System.out.println("Grupo: CINCO  Gesto: " + current.stringValue(current.classIndex()));
				}else{
					buffLabel.write(current.toString());
					buffLabel.newLine();
					buffLabel.flush();
				}
				
				insertOnHashMap(mapGroups, mapLabels, instAtual, current);
			}
			
			if(instAtual.stringValue(inst.classIndex()).equals("SEIS")){
				
				predict(mlpLabel.get(6), current, 6);
				
				if(modoConsumidor){
					System.out.println("Grupo: SEIS  Gesto: " + current.stringValue(current.classIndex()));
				}else{
					buffLabel.write(current.toString());
					buffLabel.newLine();
					buffLabel.flush();
				}
				
				insertOnHashMap(mapGroups, mapLabels, instAtual, current);
			}

			if(instAtual.stringValue(inst.classIndex()).equals("SETE")){
				
				predict(mlpLabel.get(7), current, 7);
				
				if(modoConsumidor){
					System.out.println("Grupo: SETE  Gesto: " + current.stringValue(current.classIndex()));
				}else{
					buffLabel.write(current.toString());
					buffLabel.newLine();
					buffLabel.flush();
				}
				
				insertOnHashMap(mapGroups, mapLabels, instAtual, current);
			}
			
			if(instAtual.stringValue(inst.classIndex()).equals("OITO")){
				
				predict(mlpLabel.get(8), current, 8);
				
				if(modoConsumidor){
					System.out.println("Grupo: OITO  Gesto: " + current.stringValue(current.classIndex()));
				}else{
					buffLabel.write(current.toString());
					buffLabel.newLine();
					buffLabel.flush();
				}
				
				insertOnHashMap(mapGroups, mapLabels, instAtual, current);
			}
			
			if(instAtual.stringValue(inst.classIndex()).equals("NOVE")){
				
				predict(mlpLabel.get(9), current, 9);
				
				if(modoConsumidor){
					System.out.println("Grupo: NOVE  Gesto: " + current.stringValue(current.classIndex()));
				}else{
					buffLabel.write(current.toString());
					buffLabel.newLine();
					buffLabel.flush();
				}
				
				insertOnHashMap(mapGroups, mapLabels, instAtual, current);
			}				
			
			if(instAtual.stringValue(inst.classIndex()).equals("DEZ")){
				
				predict(mlpLabel.get(10), current, 10);
				
				if(modoConsumidor){
					System.out.println("Grupo: DEZ  Gesto: " + current.stringValue(current.classIndex()));
				}else{
					buffLabel.write(current.toString());
					buffLabel.newLine();
					buffLabel.flush();
				}
				
				insertOnHashMap(mapGroups, mapLabels, instAtual, current);
			}
			
			if(instAtual.stringValue(inst.classIndex()).equals("ONZE")){
				
				predict(mlpLabel.get(11), current, 11);
				
				if(modoConsumidor){
					System.out.println("Grupo: ONZE  Gesto: " + current.stringValue(current.classIndex()));
				}else{
					buffLabel.write(current.toString());
					buffLabel.newLine();
					buffLabel.flush();
				}
				
				insertOnHashMap(mapGroups, mapLabels, instAtual, current);
			}
			
			if(!modoConsumidor){
				if(count < (dataTest.numInstances() - 1))
					count++;
				else
					break;
			}
			
			dataTest.clear();
		}
		
		if(!modoConsumidor)
			buffLabel.close();
		
		System.out.println("Salvando estatisticas...");
		
		// gerando as estatisticas
		printEstatistics(mapGroups, mapLabels);
		
		// apagar o arquivo saida_groups_predict
		writer = new BufferedWriter(new FileWriter("data/saida_groups_predict.arff"));
		dataTest = sourceTest.getDataSet();
		dataTest.deleteAttributeAt(dataTest.numAttributes() - 1);
		dataTest.delete();
		writer.write(dataTest.toString());
		writer.newLine();
		writer.flush();
		writer.close();
		
		System.out.println("Concluido!");
		
		source.reset();
		sourceTest.reset();
		dataTest.clear();
	}
	
	private static void predict(MultilayerPerceptron mlp, Instance inst, int i) throws Exception{
	
		double label = mlp.classifyInstance(inst);
	
		if(i >= 0){
			inst.setClassValue(matrixLabels[i][(int) label]);
		}else{
			inst.setClassValue(label);
		}
	}
	
	private static void printEstatistics(Map<String, Integer> mapGroups, Map<String, Integer> mapLabels) throws Exception{
		BufferedWriter buff = new BufferedWriter(new FileWriter("data/saida_resultados.dat", true));
		
		buff.write("GRUPOS");
		buff.newLine();
		buff.flush();
		for(String key : mapGroups.keySet()){
			buff.write(key + ": ");
			buff.write(mapGroups.get(key).toString());
			buff.newLine();
			buff.flush();
		}
		
		buff.write("GESTOS");
		buff.newLine();
		buff.flush();
		for(String key : mapLabels.keySet()){
			buff.write(key + ": ");
			buff.write(mapLabels.get(key).toString());
			buff.newLine();
			buff.flush();
		}
		
		buff.write("  ---------------------------------------------- ");
		buff.newLine();
		buff.flush();
		buff.close();
	}
	
	private static void insertOnHashMap(Map <String, Integer> mapGroups, Map<String, Integer> mapLabels, Instance instAtual, Instance current){
		if(mapGroups.containsKey(instAtual.stringValue(instAtual.classIndex()))){
			mapGroups.put(instAtual.stringValue(instAtual.classIndex()), mapGroups.get(instAtual.stringValue(instAtual.classIndex())) + 1);
		}else{
			mapGroups.put(instAtual.stringValue(instAtual.classIndex()), 1);
		}
		if(mapLabels.containsKey(current.stringValue(current.classIndex()))){
			mapLabels.put(current.stringValue(current.classIndex()), mapLabels.get(current.stringValue(current.classIndex())) + 1);
		}else{
			mapLabels.put(current.stringValue(current.classIndex()), 1);
		}
	}
	
}
