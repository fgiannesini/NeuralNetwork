package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation;

import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.DataContainer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

public class GradientDescentWithDerivation implements LearningAlgorithm {

    private final NeuralNetworkModel originalNeuralNetworkModel;
    private final CostType costType;
    private final double learningRate;
    private final IGradientDescentWithDerivationProcessProvider gradientDescentProcessProvider;

    public GradientDescentWithDerivation(NeuralNetworkModel neuralNetworkModel, CostType costType, double learningRate, IGradientDescentWithDerivationProcessProvider gradientDescentProcessProvider) {
        this.originalNeuralNetworkModel = neuralNetworkModel;
        this.costType = costType;
        this.learningRate = learningRate;
        this.gradientDescentProcessProvider = gradientDescentProcessProvider;
    }

    @Override
    public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
        DataContainer dataContainer = gradientDescentProcessProvider.getDataProcessLauncher().apply(new DataContainer(inputMatrix, y));
        GradientDescentWithDerivationContainer gradientDescentWithDerivationContainer = new GradientDescentWithDerivationContainer(dataContainer.getInput(), dataContainer.getOutput(), originalNeuralNetworkModel, learningRate, costType, gradientDescentProcessProvider.getCostComputerBuildingLauncher());
        return gradientDescentProcessProvider.getGradientWithDerivationLauncher().apply(gradientDescentWithDerivationContainer).getNeuralNetworkModel();
    }

    public IGradientDescentWithDerivationProcessProvider getGradientDescentProcessProvider() {
        return gradientDescentProcessProvider;
    }
}
