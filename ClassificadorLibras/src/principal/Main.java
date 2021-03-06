package principal;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.DenseInstance;
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
		DataSource source = new DataSource("data/out_hu.csv");
		
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
			DataSource ds = new DataSource("data/grupo_" + i + "_hu.csv");
			
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
		
		/*   DELETANDO ARQUIVOS DE SAIDA PARA NAO DAR ERROS ANTES DE INICIAR AS PREDICOES   */
		File fileTest = new File("data/saida_labels_predict.arff");
		if(fileTest.exists()){
			fileTest.delete();
		}
		
		fileTest = new File("data/saida_groups_predict.arff");
		if(fileTest.exists()){
			fileTest.delete();
		}
		/*  -----------------------------------------------------------------------------   */
		
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
		
		BufferedWriter buffLabel = null;
		
		// se nao for modo consumidor entao as saidas sao gravadas num arquivo
		if(!modoConsumidor){
			buffLabel = new BufferedWriter(new FileWriter("data/saida_labels_predict.arff"));
		}
		
		String strFile = "";
		if(args.length >= 2){
			strFile = args[1];
		}else{
			System.out.println("Necessario um arquivo arff para testes!\n Encerrando...");
			return;
		}
		
		DataSource sourceTest = null; // para grupos
		Instances dataTest = null; // para grupos
		
		DataSource sourceTestLabel = null; // para labels
		Instances dataTestLabel = null; // para labels
		
		// abrindo arquivo de dataset para teste
		sourceTest = new DataSource(strFile);
		sourceTestLabel = new DataSource(strFile);
		
		// deletando atributo nao util
		dataTest = sourceTest.getDataSet();
		dataTestLabel = sourceTestLabel.getDataSet();
		
		// retirando label e setando classe grupo
		dataTest.deleteAttributeAt(dataTest.numAttributes() - 1);
		dataTest.setClassIndex(dataTest.numAttributes() - 1);
		
		// retirando grupo e setando classe label
		dataTestLabel.deleteAttributeAt(dataTestLabel.numAttributes() - 2);
		dataTestLabel.setClassIndex(dataTestLabel.numAttributes() - 1);
		
		// hashmap para estatisticas
		Map<String, Integer> mapGroups = new HashMap<>();
		Map<String, Integer> mapLabels = new HashMap<>();
		
		BufferedWriter writer;
		File fileGroupsPredict = new File("data/saida_groups_predict.arff");
		File fileDataset = new File(strFile);
		
		if(!fileGroupsPredict.exists()){
			writer = new BufferedWriter(new FileWriter("data/saida_groups_predict.arff"));
			
			dataTest.delete();
			
			writer.write(dataTest.toString());
			writer.newLine();
		 	writer.flush();
			writer.close();
		 	
		 	dataTest = sourceTest.getDataSet();
			dataTest.deleteAttributeAt(dataTest.numAttributes() - 1);
			dataTest.setClassIndex(dataTest.numAttributes() - 1);
			
		}
		
		// pegando instancias com os grupos preditos
		DataSource ds = new DataSource("data/saida_groups_predict.arff");
		Instances inst = ds.getDataSet();
		
		// classe grupos esta na ultima posicao
		inst.setClassIndex(inst.numAttributes() - 1);
		
		// abrindo para modo de escrita 'append'
		writer = new BufferedWriter(new FileWriter("data/saida_groups_predict.arff", true));
		
		Instance atualGrupo = null; // ultima instancia do arquivo de testes com classe grupo
		Instance atualLabel = null; // ultima instancia do arquivo de testes com classe label
		Instance instAtual = null; // ultima instancia do arquivo saida_groups_predict
		
		System.out.println("Iniciando predicao...");
		
		Scanner scan = new Scanner(System.in);
		
		char continuaExecucao = 's';
		while(continuaExecucao == 's'){
			
			// roda indefinidamente
			while(true){
				
				// se for modo consumidor entao testa se arquivo de tempo real existe
				if(modoConsumidor){
					if(!fileDataset.exists()){
						System.out.println("Esperando pelo arquivo...");
						Thread.sleep(100);
						continue;
					}else{
						// resgistra tempo de modificacao
						if(tempoAnterior == 0){
							tempoAnterior = fileDataset.lastModified();
						}else{
							if(tempoAnterior == fileDataset.lastModified()){
								System.out.println("Nenhuma entrada nova no arquivo...");
								Thread.sleep(100);
								continue;
							}else{
								count++;
								tempoAnterior = fileDataset.lastModified();
							}
						}
					}
				}else{
					if(!fileDataset.exists()){
						System.out.println("Arquivo de testes nao existe! Criar arquivo: " + fileDataset.getName());
						continue;
					}
				}
				
				// pequeno arranjo tecnico
				if(count == 0 && modoConsumidor){
					continue;
				}
				
				// adiciona a dataTest e a dataTestLabel instancia lida do arquivo de testes
				if(modoConsumidor){
					Instance c = instanceFromFile(fileDataset, 0, null);
					dataTest.add(c);
					dataTestLabel.add(c);
				}
				
				// se for modo consumidor, pega a ultima - se nao, pega na posicao count
				if(modoConsumidor){
					if(dataTest.numInstances() >= 1){
						atualGrupo = dataTest.lastInstance();
					}else{
						System.out.println("Nenhuma instancia inserida. Aguardando...");
						Thread.sleep(10000);
						continue;
					}
				}else{
					// se tiver pelo menos uma instancia ele faz o teste - para grupo
					if(dataTest.numInstances() >=1){
						atualGrupo = dataTest.instance(count);
					}else{
						System.out.println("Arquivo de instancias para predict vazio. Encerrando...");
						return;
					}
					
				}
				
				// efetua o predict para a instancia 'atual' utilizando a MLP dos grupos
				predict(mlpGrupos, atualGrupo, -1);
					
				// escreve no arquivo saida_grupos_predict a instancia que acabou de ser predita com sua classe
				writer.write(atualGrupo.toString());
			 	writer.newLine();
				writer.flush();
				
				if(modoConsumidor){
					atualLabel = dataTestLabel.lastInstance();
				}else{
					atualLabel = dataTestLabel.instance(count);
				}
				
				// sempre adiciona a inst (saida_groups_predict) e sempre pega a ultima predita
				inst.add(instanceFromFile(new File("data/saida_groups_predict.arff"), 1, inst));
				instAtual = inst.lastInstance();
				
				// transforma o label de instAtual.stringValue(inst.classIndex()) em um valor inteiro de 0 a 11
				int strInt = strToInt(instAtual.stringValue(inst.classIndex()));
				
				if(strInt < 0){
					System.out.println("Erro na funcao strInt. Valor retornado: " + strInt);
					System.out.println("Encerrando...");
					return;
				}
				
				// de acordo com o grupo retornado, chama a MLP label correspondente e passa a instancia current e seu grupo
				// strInt varia de 0 a 11 e representa o grupo (linhas da variavel global matrixLabels
				predict(mlpLabel.get(strInt), atualLabel, strInt);
				
				if(modoConsumidor){
					System.out.println("Grupo: " + instAtual.stringValue(inst.classIndex()) + "    Gesto: " + atualLabel.stringValue(atualLabel.classIndex()));
				}else{
					buffLabel.write(atualLabel.toString());
					buffLabel.newLine();
					buffLabel.flush();
				}
				
				// insere nas hashmaps para estatisticas o grupo e label que foram preditos pelas MLPs
				insertOnHashMap(mapGroups, mapLabels, instAtual, atualLabel);
				
				// se esta no modo consumidor, entao devera incrementar count
				if(!modoConsumidor){
					if(count < (dataTest.numInstances() - 1))
						count++;
					else
						break;
				}
			}
			
			System.out.println("Salvando estatisticas...");
			
			// gerando as estatisticas
			printEstatistics(mapGroups, mapLabels);

			strFile = hasMoreFile(scan);
			
			if(strFile.equals("")){
				continuaExecucao = 'n';
			}else{
				// limpa as variaveis para processar o proximo arquivo e efetuar os predicts
				
				dataTest.delete();
				dataTestLabel.delete();
				
				sourceTest.reset();
				sourceTestLabel.reset();
				
				// resetando dataset de saida_groups_predict
				inst.delete();
				
				count = 0;
				
				mapGroups.clear();
				mapLabels.clear();
				
				// abrindo arquivo de dataset para teste
				sourceTest = new DataSource(strFile);
				sourceTestLabel = new DataSource(strFile);
				
				// deletando atributo nao util
				dataTest = sourceTest.getDataSet();
				dataTestLabel = sourceTestLabel.getDataSet();
				
				// retirando label e setando classe grupo
				dataTest.deleteAttributeAt(dataTest.numAttributes() - 1);
				dataTest.setClassIndex(dataTest.numAttributes() - 1);
				
				// retirando grupo e setando classe label
				dataTestLabel.deleteAttributeAt(dataTestLabel.numAttributes() - 2);
				dataTestLabel.setClassIndex(dataTestLabel.numAttributes() - 1);
			}
		}
		
		if(!modoConsumidor)
			buffLabel.close();
		
		writer.close();
		
		// apagar o arquivo saida_groups_predict e refazer cabe�alho
		writer = new BufferedWriter(new FileWriter("data/saida_groups_predict.arff"));
		dataTest = sourceTest.getDataSet();
		dataTest.deleteAttributeAt(dataTest.numAttributes() - 1);
		dataTest.delete();
		writer.write(dataTest.toString());
		writer.newLine();
		writer.flush();
		writer.close();
		
		System.out.println("Concluido!");
		
		scan.close();
		
		source.reset();
		sourceTest.reset();
		dataTest.clear();
		dataTestLabel.clear();
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
			buff.write("\t" + key + ": ");
			buff.write(mapGroups.get(key).toString());
			buff.newLine();
			buff.flush();
		}
		
		buff.write("GESTOS");
		buff.newLine();
		buff.flush();
		for(String key : mapLabels.keySet()){
			buff.write("\t" + key + ": ");
			buff.write(mapLabels.get(key).toString());
			buff.newLine();
			buff.flush();
		}
		
		buff.write("----------------------------------------------");
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
	
	private static int strToInt(String value){
		
		switch (value) {
		case "ZERO":
			return 0;
			
		case "UM":
			return 1;
		
		case "DOIS":
			return 2;
			
		case "TRES":
			return 3;
		
		case "QUATRO":
			return 4;
		
		case "CINCO":
			return 5;
		
		case "SEIS":
			return 6;
		
		case "SETE":
			return 7;
		
		case "OITO":
			return 8;
		
		case "NOVE":
			return 9;
		
		case "DEZ":
			return 10;
		
		case "ONZE":
			return 11;
			
		default:
			System.out.println("Erro na funcao strToInt!");
			return -1;
		}

	}
	
	private static Instance instanceFromFile(File file, int type, Instances dataset) throws IOException{
		
		BufferedReader br = new BufferedReader(new FileReader(file));
		
		String lastLine = "";
		String sCurrentLine = "";
		
	    while ((sCurrentLine = br.readLine()) != null) 
	    {
	        lastLine = sCurrentLine;
	    }
	    
	    br.close();
	    
	    String[] values = lastLine.split(",");
	    
	    Instance inst = new DenseInstance(values.length);
	    
	    if(type == 1){
	    	for(int i = 0; i < values.length - 1; i++){
		    	inst.setValue(i, Double.parseDouble(values[i]));
		    }
	    	inst.setDataset(dataset);
	    	inst.setValue((values.length - 1), String.valueOf(values[values.length - 1]));
	    	
	    	return inst;
	    }
	    
	    for(int i = 0; i < values.length; i++){
	    	if(!values[i].equals("?"))
	    		inst.setValue(i, Double.parseDouble(values[i]));
	    }
	    
    	inst.deleteAttributeAt(inst.numAttributes() - 1);
    	inst.setMissing(inst.numAttributes() - 1);
	
		return inst;
	}
	
	private static String hasMoreFile(Scanner scan){
		String str;
		while(true){
			System.out.println("Deseja processar mais algum arquivo? <s|n>");
			str = scan.next();
			if(str.equals("s") || str.equals("n")){
				if(str.equals("s")){
					System.out.println("Entre com o nome do arquivo: ");
					str = scan.next();
					if(new File(str).exists()){
						return str;
					}else{
						System.out.println("Arquivo nao existe! Entre com um arquivo valido ou <n> para encerrar!");
						continue;
					}
				}else{
					if(str.equals("n")){
						return "";
					}
				}
				return "";
			}else{
				System.out.println("Entrada invalida! Use <s|n> para continuar ou nao.");
				continue;
			}
		}
	}
}
