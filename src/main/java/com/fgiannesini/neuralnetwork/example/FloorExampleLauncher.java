package com.fgiannesini.neuralnetwork.example;

import com.fgiannesini.neuralnetwork.*;
import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
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
    private static int INPUT_COUNT = 100_000;
    private static int TEST_INPUT_COUNT = 1_000;

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

        neuralNetwork.learn(inputMatrix, outputMatrix, testInputMatrix, testOutputMatrix);

        DoubleMatrix testOutputPredictionMatrix = neuralNetwork.apply(testInputMatrix);

        return ExampleDataManager.computeSuccessRate(testOutputMatrix, testOutputPredictionMatrix);
    }

    private NeuralNetwork prepare() {
        NeuralNetworkModelBuilder neuralNetworkModelBuilder = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.XAVIER)
                .input(1);

        int[] hiddenLayerSize = hyperParameters.getHiddenLayerSize();
        for (int hiddenLayerIndex : hiddenLayerSize) {
            neuralNetworkModelBuilder.addLayer(hiddenLayerIndex, ActivationFunctionType.RELU);
        }
        neuralNetworkModelBuilder.addLayer(10, ActivationFunctionType.SOFT_MAX);

        NeuralNetworkModel neuralNetworkModel;
        if (hyperParameters.getLayerType() == LayerType.WEIGHT_BIAS) {
            neuralNetworkModel = neuralNetworkModelBuilder.buildWeightBiasModel();
        } else {
            neuralNetworkModel = neuralNetworkModelBuilder.buildBatchNormModel();
        }

        return NeuralNetworkBuilder.init()
                .withNeuralNetworkModel(neuralNetworkModel)
                .withLearningAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT)
                .withCostType(CostType.SOFT_MAX_REGRESSION)
                .withNeuralNetworkStatsConsumer(statsUpdateAction)
                .withHyperParameters(hyperParameters)
                .withNormalizer(NormalizerType.MEAN_AND_DEVIATION.get())
                .build();
    }

}
