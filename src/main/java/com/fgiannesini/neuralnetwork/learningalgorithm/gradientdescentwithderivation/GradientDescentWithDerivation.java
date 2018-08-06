package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation;

import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.function.Function;

public class GradientDescentWithDerivation implements LearningAlgorithm {

    private final NeuralNetworkModel originalNeuralNetworkModel;
    private final CostType costType;
    private final double learningRate;
    private final IGradientDescentWithDerivationProcessProvider gradientDescentProcessProvider;

    public GradientDescentWithDerivation(NeuralNetworkModel neuralNetworkModel, CostType costType, double learningRate) {
        this.originalNeuralNetworkModel = neuralNetworkModel;
        this.costType = costType;
        this.learningRate = learningRate;
        gradientDescentProcessProvider = new GradientDescentWithDerivationProcessProvider();
    }

    @Override
    public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
        GradientDescentWithDerivationContainer gradientDescentWithDerivationContainer = new GradientDescentWithDerivationContainer(inputMatrix, y, originalNeuralNetworkModel, learningRate, costType, gradientDescentProcessProvider.getCostComputerBuildingLauncher());
        Function<GradientDescentWithDerivationContainer, GradientDescentWithDerivationContainer> gradientDescentWithDerivationContainerFunction = gradientDescentProcessProvider.getGradientWithDerivationLauncher();
        return gradientDescentWithDerivationContainerFunction.apply(gradientDescentWithDerivationContainer).getNeuralNetworkModel();
    }
}
