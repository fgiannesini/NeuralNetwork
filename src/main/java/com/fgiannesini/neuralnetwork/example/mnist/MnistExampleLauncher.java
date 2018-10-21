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
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.normalizer.NormalizerType;
import org.jblas.DoubleMatrix;

import java.awt.image.BufferedImage;
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
                .batchSize(50)
                .epochCount(1)
                .momentumCoeff(null)
                .rmsStopCoeff(null)
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
                    final BufferedImage image = testMnistReader.getDataAsBufferedImage(data);
                    DoubleMatrix inputMatrix = new DoubleMatrix(image.getHeight(), image.getWidth());
                    for (int i = 0; i < image.getWidth(); i++) {
                        for (int j = 0; j < image.getHeight(); j++) {
                            inputMatrix.put(j, i, image.getRGB(i, j));
                        }
                    }
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
        mnistReader.handleSome(6_000,
                (index, data, item) -> {
                    final BufferedImage image = mnistReader.getDataAsBufferedImage(data);
                    DoubleMatrix inputMatrix = new DoubleMatrix(image.getHeight(), image.getWidth());
                    for (int i = 0; i < image.getWidth(); i++) {
                        for (int j = 0; j < image.getHeight(); j++) {
                            inputMatrix.put(j, i, image.getRGB(i, j));
                        }
                    }
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
                null,
                null
        );

        LayerTypeData testOutputPredictionMatrix = neuralNetwork.apply(inputTestData);
        DataExtractorVisitor dataVisitor = new DataExtractorVisitor();
        testOutputPredictionMatrix.accept(dataVisitor);
        return ExampleDataManager.computeSuccessRate(outputTestData.getData(), dataVisitor.getData());
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
                .input(28, 28, 1)
                .addConvolutionLayer(5, 0, 1, 32, ActivationFunctionType.RELU)
                .addMaxPoolingLayer(2, 0, 2, ActivationFunctionType.RELU)
                .addConvolutionLayer(5, 0, 1, 64, ActivationFunctionType.RELU)
                .addMaxPoolingLayer(2, 0, 2, ActivationFunctionType.RELU)
                .addFullyConnectedLayer(10, ActivationFunctionType.SOFT_MAX);

        NeuralNetworkModel neuralNetworkModel = neuralNetworkModelBuilder.buildConvolutionNetworkModel();
        return NeuralNetworkBuilder.init()
                .withNeuralNetworkModel(neuralNetworkModel)
                .withLearningAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT)
                .withCostType(CostType.SOFT_MAX_REGRESSION)
                .withNeuralNetworkStatsConsumer(statsUpdateAction)
                .withHyperParameters(hyperParameters)
                .withNormalizer(NormalizerType.NONE.get())
                .build();
    }
}
