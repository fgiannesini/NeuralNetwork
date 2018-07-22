package com.fgiannesini.neuralnetwork.example;

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

    public FloorExampleLauncher(Consumer<NeuralNetworkStats> statsUpdateAction) {
        this.statsUpdateAction = statsUpdateAction;
    }

    public static void main(String[] args) {
        Consumer<NeuralNetworkStats> statsUpdateAction = neuralNetworkStats -> {
            System.out.println("Batch number " + neuralNetworkStats.getBatchNumber());
            System.out.println("Iteration number " + neuralNetworkStats.getIterationNumber());
            System.out.println("LearningCost = " + neuralNetworkStats.getLearningCost());
            System.out.println("TestCost = " + neuralNetworkStats.getTestCost());
            System.out.println();
        };
        FloorExampleLauncher floorExampleLauncher = new FloorExampleLauncher(statsUpdateAction);
        double successRate = floorExampleLauncher.launch();
        System.out.println("Success Rate: " + successRate + "%");
    }

    public double launch() {
        NeuralNetwork neuralNetwork = prepare();

        int learningSize = 100_000;
        DoubleMatrix inputMatrix = ExampleDataManager.generateInputData(learningSize);
        DoubleMatrix outputMatrix = ExampleDataManager.convertToOutputFormat(inputMatrix.data);

        int testSize = 100;
        DoubleMatrix testInputMatrix = ExampleDataManager.generateInputData(testSize);
        DoubleMatrix testOutputMatrix = ExampleDataManager.convertToOutputFormat(testInputMatrix.data);

        neuralNetwork.learn(inputMatrix, outputMatrix, testInputMatrix, testOutputMatrix);

        DoubleMatrix testOutputPredictionMatrix = neuralNetwork.apply(testInputMatrix);

        return ExampleDataManager.computeSuccessRate(testOutputMatrix, testOutputPredictionMatrix);
    }

    private NeuralNetwork prepare() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.XAVIER)
                .input(1)
                .addLayer(10, ActivationFunctionType.RELU)
                .addLayer(10, ActivationFunctionType.SIGMOID)
                .build();

        return NeuralNetworkBuilder.init()
                .withNeuralNetworkModel(neuralNetworkModel)
                .withCostType(CostType.LOGISTIC_REGRESSION)
                .withNeuralNetworkStatsConsumer(statsUpdateAction)
                .build();
    }

}
