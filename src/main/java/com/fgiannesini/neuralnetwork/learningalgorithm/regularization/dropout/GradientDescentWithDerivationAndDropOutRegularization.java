package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.DataContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivationContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.IGradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

public class GradientDescentWithDerivationAndDropOutRegularization implements LearningAlgorithm {

    private final NeuralNetworkModel neuralNetworkModel;
    private final CostType costType;
    private final double learningRate;
    private IGradientDescentWithDerivationProcessProvider gradientDescentWithDerivationProcessProvider;

    @Override
    public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
        DataContainer dataContainer = gradientDescentWithDerivationProcessProvider.getDataProcessLauncher().apply(new DataContainer(inputMatrix,y));
        return gradientDescentWithDerivationProcessProvider.getGradientWithDerivationLauncher().apply(new GradientDescentWithDerivationContainer(dataContainer.getInput(), dataContainer.getOutput(), neuralNetworkModel
                , learningRate, costType, gradientDescentWithDerivationProcessProvider.getCostComputerBuildingLauncher())).getNeuralNetworkModel();
    }
}
