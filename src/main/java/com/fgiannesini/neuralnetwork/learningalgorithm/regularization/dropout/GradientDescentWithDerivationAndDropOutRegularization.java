package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivationContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.IGradientDescentWithDerivationProcessProvider;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.function.Supplier;

public class GradientDescentWithDerivationAndDropOutRegularization implements LearningAlgorithm {

    private final NeuralNetworkModel neuralNetworkModel;
    private final CostType costType;
    private final Supplier<List<DoubleMatrix>> dropOutMatricesSupplier;
    private final double learningRate;

    public GradientDescentWithDerivationAndDropOutRegularization(NeuralNetworkModel neuralNetworkModel, CostType costType, double learningRate, Supplier<List<DoubleMatrix>> dropOutMatricesSupplier) {
        this.neuralNetworkModel = neuralNetworkModel;
        this.costType = costType;
        this.dropOutMatricesSupplier = dropOutMatricesSupplier;
        this.learningRate = learningRate;
    }

    @Override
    public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
        List<DoubleMatrix> dropOutMatrices = dropOutMatricesSupplier.get();
        IGradientDescentWithDerivationProcessProvider gradientDescentWithDerivationProcessProvider = new GradientDescentWithDerivationandDropOutRegularizationProcessProvider(dropOutMatrices);
        DoubleMatrix dropOutOutput = y.mulColumnVector(dropOutMatrices.get(dropOutMatrices.size() - 1));
        return gradientDescentWithDerivationProcessProvider.getGradientWithDerivationLauncher().apply(new GradientDescentWithDerivationContainer(inputMatrix, dropOutOutput, neuralNetworkModel
                , learningRate, costType, gradientDescentWithDerivationProcessProvider.getCostComputerBuildingLauncher())).getNeuralNetworkModel();
    }
}
