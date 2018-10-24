package com.fgiannesini.neuralnetwork.example.mnist;

import com.fgiannesini.neuralnetwork.*;
import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.computer.data.ConvolutionData;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.example.floor.DataExtractorVisitor;
import com.fgiannesini.neuralnetwork.example.floor.ExampleDataManager;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithmType;
import com.fgiannesini.neuralnetwork.learningrate.LearningRateUpdaterType;
import com.fgiannesini.neuralnetwork.model.ConvolutionNeuralNetworkModelBuilder;
import com.fgiannesini.neuralnetwork.model.LayerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.normalizer.NormalizerType;
import com.fgiannesini.neuralnetwork.normalizer.meandeviation.MeanDeviationProvider;
import org.jblas.DoubleMatrix;

import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Consumer;

public class MnistExampleLauncher {

    private final Consumer<NeuralNetworkStats> statsUpdateAction;
    private final HyperParameters hyperParameters;

    public MnistExampleLauncher(Consumer<NeuralNetworkStats> statsUpdateAction, HyperParameters hyperParameters) {
        this.statsUpdateAction = statsUpdateAction;
        this.hyperParameters = hyperParameters;
    }

    public static void main(String[] args) throws IOException {
        Consumer<NeuralNetworkStats> statsUpdateAction = neuralNetworkStats -> {
            System.out.println("Batch number " + neuralNetworkStats.getBatchNumber());
            System.out.println("Epoch number " + neuralNetworkStats.getEpochNumber());
            System.out.println("LearningCost = " + neuralNetworkStats.getLearningCost());
            System.out.println("TestCost = " + neuralNetworkStats.getTestCost());
            System.out.println();
        };
        HyperParameters parameters = new HyperParameters()
                .learningRateUpdater(LearningRateUpdaterType.CONSTANT.get(0.01))
                .batchSize(100)
                .epochCount(1)
                .momentumCoeff(null)
                .rmsStopCoeff(null)
                .layerType(LayerType.POOLING_MAX)
                .hiddenLayerSize(new int[0])
                .convolutionLayers(new int[]{16, 8})
                .regularizationCoeff(new RegularizationCoeffs());
        MnistExampleLauncher mnistExampleLauncher = new MnistExampleLauncher(statsUpdateAction, parameters);
        double successRate = mnistExampleLauncher.launch();
        System.out.println("Success Rate: " + successRate + "%");
    }

    public double launch() throws IOException {
        NeuralNetwork neuralNetwork = prepare();

        MnistReader testMnistReader = new MnistReader(getFile("t10k-labels.idx1-ubyte"), getFile("t10k-images.idx3-ubyte"));
        List<DoubleMatrix> testInputMatrices = new ArrayList<>();
        List<Integer> testOutput = new ArrayList<>();
        testMnistReader.handleSome(1_000,
                (index, data, item) -> {
                    DoubleMatrix inputMatrix = convertDataToDoubleMatrix(testMnistReader, data);
                    testInputMatrices.add(inputMatrix);
                    testOutput.add((int) item);
                }
        );
        WeightBiasData outputTestData = new WeightBiasData(ExampleDataManager.convertToOutputFormat(testOutput.stream().mapToInt(Integer::intValue).toArray()));
        ConvolutionData inputTestData = new ConvolutionData(testInputMatrices, 1);
        testMnistReader.close();

        MnistReader mnistReader = new MnistReader(getFile("train-labels.idx1-ubyte"), getFile("train-images.idx3-ubyte"));
        List<DoubleMatrix> inputMatrices = new ArrayList<>();
        List<Integer> output = new ArrayList<>();
        mnistReader.handleSome(10_000,
                (index, data, item) -> {
                    DoubleMatrix inputMatrix = convertDataToDoubleMatrix(mnistReader, data);
                    inputMatrices.add(inputMatrix);
                    output.add((int) item);
                }
        );
        WeightBiasData outputData = new WeightBiasData(ExampleDataManager.convertToOutputFormat(output.stream().mapToInt(Integer::intValue).toArray()));
        ConvolutionData inputData = new ConvolutionData(inputMatrices, 1);
        mnistReader.close();

        neuralNetwork.learn(
                inputData,
                outputData,
                inputTestData,
                outputTestData
        );

        LayerTypeData testOutputPredictionMatrix = neuralNetwork.apply(inputTestData);
        DataExtractorVisitor dataVisitor = new DataExtractorVisitor();
        testOutputPredictionMatrix.accept(dataVisitor);
        return ExampleDataManager.computeSuccessRate(outputTestData.getData(), dataVisitor.getData());
    }

    private DoubleMatrix convertDataToDoubleMatrix(MnistReader mnistReader, byte[] data) {
        double[] grayData = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            int gray = 255 - (((int) data[i]) & 0xFF);
            grayData[i] = gray;
        }
        return new DoubleMatrix(mnistReader.getImageHeight(), mnistReader.getImageWidth(), grayData);
    }

    public File getFile(String fileName) {
        try {
            return new File(this.getClass().getClassLoader().getResource("mnist/" + fileName).toURI());
        } catch (URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    private NeuralNetwork prepare() {
        ConvolutionNeuralNetworkModelBuilder neuralNetworkModelBuilder = ConvolutionNeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.XAVIER)
                .input(28, 28, 1);
        int[] convolutionLayers = hyperParameters.getConvolutionLayers();
        for (int convolutionLayer : convolutionLayers) {
            neuralNetworkModelBuilder.addConvolutionLayer(5, 0, 1, convolutionLayer, ActivationFunctionType.RELU);
//            if (hyperParameters.getLayerType().equals(LayerType.POOLING_AVERAGE)) {
//                neuralNetworkModelBuilder.addAveragePoolingLayer(2, 0, 2, ActivationFunctionType.RELU);
//            } else {
//                neuralNetworkModelBuilder.addMaxPoolingLayer(2, 0, 2, ActivationFunctionType.RELU);
//            }
        }

        int[] hiddenLayerSize = hyperParameters.getHiddenLayerSize();
        for (int i = 0; i < hiddenLayerSize.length - 1; i++) {
            neuralNetworkModelBuilder.addFullyConnectedLayer(hiddenLayerSize[i], ActivationFunctionType.RELU);
        }
        neuralNetworkModelBuilder.addFullyConnectedLayer(10, ActivationFunctionType.SOFT_MAX);

        NeuralNetworkModel neuralNetworkModel = neuralNetworkModelBuilder.buildConvolutionNetworkModel();
        return NeuralNetworkBuilder.init()
                .withNeuralNetworkModel(neuralNetworkModel)
                .withLearningAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT)
                .withCostType(CostType.SOFT_MAX_REGRESSION)
                .withNeuralNetworkStatsConsumer(statsUpdateAction)
                .withHyperParameters(hyperParameters)
                .withNormalizer(NormalizerType.MEAN_AND_DEVIATION.get(new MeanDeviationProvider()))
                .build();
    }
}
