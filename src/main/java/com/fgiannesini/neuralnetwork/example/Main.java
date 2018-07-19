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

import java.util.Observable;
import java.util.Observer;

public class Main {

    public static void main(String[] args) {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .useInitializer(InitializerType.XAVIER)
                .input(1)
                .addLayer(10, ActivationFunctionType.RELU)
                .addLayer(10, ActivationFunctionType.TANH)
                .build();

        NeuralNetwork neuralNetwork = NeuralNetworkBuilder.init()
                .withNeuralNetworkModel(neuralNetworkModel)
                .withCostType(CostType.LOGISTIC_REGRESSION)
                .build();

        Observable statsObservable = neuralNetwork.getStatsObservable();
        statsObservable.addObserver(new Observer() {
            @Override
            public void update(Observable o, Object arg) {
                NeuralNetworkStats neuralNetworkStats = (NeuralNetworkStats) arg;
                System.out.println("learningCost = " + neuralNetworkStats.getLearningCost());
                System.out.println("testCost = " + neuralNetworkStats.getTestCost());
                System.out.println();
            }
        });

        int learningSize = 1000;
        DoubleMatrix inputMatrix = ExampleDataManager.generateInputData(learningSize);
        DoubleMatrix outputMatrix = ExampleDataManager.convertToOutputFormat(inputMatrix.data);

        int testSize = 100;
        DoubleMatrix testInputMatrix = ExampleDataManager.generateInputData(testSize);
        DoubleMatrix testOutputMatrix = ExampleDataManager.convertToOutputFormat(testInputMatrix.data);

        neuralNetwork.learn(inputMatrix, outputMatrix, testInputMatrix, testOutputMatrix);

        DoubleMatrix testOutputPredictionMatrix = neuralNetwork.apply(testInputMatrix);

        double successRate = ExampleDataManager.computeSuccessRate(testOutputMatrix, testOutputPredictionMatrix);
        System.out.println("Success Rate = " + successRate + " %");
    }

}
