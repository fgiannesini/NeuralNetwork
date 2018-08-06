package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2;

import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivationContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.IGradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

public class GradientDescentWithDerivationAndL2Regularization implements LearningAlgorithm {

    private final NeuralNetworkModel neuralNetworkModel;
    private final CostType costType;
    private final double learningRate;
    private final IGradientDescentWithDerivationProcessProvider gradientDescentWithDerivationProcessProvider;

    public GradientDescentWithDerivationAndL2Regularization(NeuralNetworkModel neuralNetworkModel, CostType costType, double learningRate, double regularizationCoeff) {
        this.neuralNetworkModel = neuralNetworkModel;
        this.costType = costType;
        this.learningRate = learningRate;
        gradientDescentWithDerivationProcessProvider = new GradientDescentWithDerivationAndL2RegularizationProcessProvider(regularizationCoeff);
    }


    @Override
    public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
        return gradientDescentWithDerivationProcessProvider.getGradientWithDerivationLauncher().apply(new GradientDescentWithDerivationContainer(inputMatrix, y, neuralNetworkModel, learningRate, costType,gradientDescentWithDerivationProcessProvider.getCostComputerBuildingLauncher())).getNeuralNetworkModel();
    }
}
