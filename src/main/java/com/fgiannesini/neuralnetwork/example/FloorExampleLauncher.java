package com.fgiannesini.neuralnetwork.example;

import com.fgiannesini.neuralnetwork.HyperParameters;
import com.fgiannesini.neuralnetwork.NeuralNetwork;
import com.fgiannesini.neuralnetwork.NeuralNetworkBuilder;
import com.fgiannesini.neuralnetwork.NeuralNetworkStats;
import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.initializer.InitializerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;

import java.util.function.Consumer;

public class FloorExampleLauncher {

    private final Consumer<NeuralNetworkStats> statsUpdateAction;
    private final HyperParameters hyperParameters;

    public FloorExampleLauncher(Consumer<NeuralNetworkStats> statsUpdateAction, HyperParameters hyperParameters) {
        this.statsUpdateAction = statsUpdateAction;
        this.hyperParameters = hyperParameters;
    }

    public static void main(String[] args) {
        Consumer<NeuralNetworkStats> statsUpdateAction = neuralNetworkStats -> {
            System.out.println("Batch number " + neuralNetworkStats.getBatchNumber());
            System.out.println("Iteration number " + neuralNetworkStats.getIterationNumber());
            System.out.println("LearningCost = " + neuralNetworkStats.getLearningCost());
            System.out.println("TestCost = " + neuralNetworkStats.getTestCost());
            System.out.println();
        };
        HyperParameters parameters = new HyperParameters();
        FloorExampleLauncher floorExampleLauncher = new FloorExampleLauncher(statsUpdateAction, parameters);
        double successRate = floorExampleLauncher.launch();
        System.out.println("Success Rate: " + successRate + "%");
    }

    public double launch() {
        NeuralNetwork neuralNetwork = prepare();

        DoubleMatrix inputMatrix = ExampleDataManager.generateInputData(hyperParameters.getInputCount());
        DoubleMatrix outputMatrix = ExampleDataManager.convertToOutputFormat(inputMatrix.data);

        DoubleMatrix testInputMatrix = ExampleDataManager.generateInputData(hyperParameters.getTestInputCount());
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
        neuralNetworkModelBuilder.addLayer(10, ActivationFunctionType.SIGMOID);
        NeuralNetworkModel neuralNetworkModel = neuralNetworkModelBuilder.build();

        return NeuralNetworkBuilder.init()
                .withNeuralNetworkModel(neuralNetworkModel)
                .withCostType(CostType.LOGISTIC_REGRESSION)
                .withNeuralNetworkStatsConsumer(statsUpdateAction)
                .withHyperParameters(hyperParameters)
                .build();
    }

}
