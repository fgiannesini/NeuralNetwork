package com.fgiannesini.neuralnetwork.example;

import com.fgiannesini.neuralnetwork.*;
import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.MeanDeviationProvider;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithmType;
import com.fgiannesini.neuralnetwork.learningrate.LearningRateUpdaterType;
import com.fgiannesini.neuralnetwork.model.LayerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import com.fgiannesini.neuralnetwork.normalizer.NormalizerType;
import org.jblas.DoubleMatrix;

import java.util.function.Consumer;

public class FloorExampleLauncher {

    private final Consumer<NeuralNetworkStats> statsUpdateAction;
    private final HyperParameters hyperParameters;
    private static final int INPUT_COUNT = 100_000;
    private static final int TEST_INPUT_COUNT = 1_000;

    public FloorExampleLauncher(Consumer<NeuralNetworkStats> statsUpdateAction, HyperParameters hyperParameters) {
        this.statsUpdateAction = statsUpdateAction;
        this.hyperParameters = hyperParameters;
    }

    public static void main(String[] args) {
        Consumer<NeuralNetworkStats> statsUpdateAction = neuralNetworkStats -> {
            System.out.println("Batch number " + neuralNetworkStats.getBatchNumber());
            System.out.println("Epoch number " + neuralNetworkStats.getEpochNumber());
            System.out.println("LearningCost = " + neuralNetworkStats.getLearningCost());
            System.out.println("TestCost = " + neuralNetworkStats.getTestCost());
            System.out.println();
        };
        HyperParameters parameters = new HyperParameters()
                .learningRateUpdater(LearningRateUpdaterType.CONSTANT.get(0.01))
                .batchSize(1_000)
                .epochCount(20)
                .hiddenLayerSize(new int[]{20})
                .momentumCoeff(0.9)
                .rmsStopCoeff(0.999)
                .layerType(LayerType.WEIGHT_BIAS)
                .regularizationCoeff(new RegularizationCoeffs());
        FloorExampleLauncher floorExampleLauncher = new FloorExampleLauncher(statsUpdateAction, parameters);
        double successRate = floorExampleLauncher.launch();
        System.out.println("Success Rate: " + successRate + "%");
    }

    public double launch() {
        NeuralNetwork neuralNetwork = prepare();

        DoubleMatrix inputMatrix = ExampleDataManager.generateInputData(INPUT_COUNT);
        DoubleMatrix outputMatrix = ExampleDataManager.convertToOutputFormat(inputMatrix.data);

        DoubleMatrix testInputMatrix = ExampleDataManager.generateInputData(TEST_INPUT_COUNT);
        DoubleMatrix testOutputMatrix = ExampleDataManager.convertToOutputFormat(testInputMatrix.data);

        LayerTypeData testInput = getData(testInputMatrix, hyperParameters.getLayerType());
        neuralNetwork.learn(
                getData(inputMatrix, hyperParameters.getLayerType()),
                getData(outputMatrix, hyperParameters.getLayerType()),
                testInput,
                getData(testOutputMatrix, hyperParameters.getLayerType())
        );

        LayerTypeData testOutputPredictionMatrix = neuralNetwork.apply(testInput);
        DataExtractorVisitor dataVisitor = new DataExtractorVisitor();
        testOutputPredictionMatrix.accept(dataVisitor);
        return ExampleDataManager.computeSuccessRate(testOutputMatrix, dataVisitor.getData());
    }

    private LayerTypeData getData(DoubleMatrix matrix, LayerType layerType) {
        switch (layerType) {
            case WEIGHT_BIAS:
                return new WeightBiasData(matrix);
            case BATCH_NORM:
                return new BatchNormData(matrix, new MeanDeviationProvider());
            default:
                throw new RuntimeException(layerType + " not managed");
        }
    }

    private NeuralNetwork prepare() {
        NeuralNetworkModelBuilder neuralNetworkModelBuilder = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.XAVIER)
                .input(1);

        int[] hiddenLayerSize = hyperParameters.getHiddenLayerSize();
        for (int hiddenLayerIndex : hiddenLayerSize) {
            if (hyperParameters.getLayerType() == LayerType.WEIGHT_BIAS) {
                neuralNetworkModelBuilder.addWeightBiasLayer(hiddenLayerIndex, ActivationFunctionType.RELU);
            } else {
                neuralNetworkModelBuilder.addBatchNormLayer(hiddenLayerIndex, ActivationFunctionType.RELU);
            }
        }
        if (hyperParameters.getLayerType() == LayerType.WEIGHT_BIAS) {
            neuralNetworkModelBuilder.addWeightBiasLayer(10, ActivationFunctionType.SOFT_MAX);
        } else {
            neuralNetworkModelBuilder.addBatchNormLayer(10, ActivationFunctionType.SOFT_MAX);
        }

        NeuralNetworkModel neuralNetworkModel = neuralNetworkModelBuilder.buildNeuralNetworkModel();
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
