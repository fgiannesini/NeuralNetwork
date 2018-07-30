package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.*;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.function.Supplier;

public class GradientDescentWithDropOutRegularization implements LearningAlgorithm {

    private List<DoubleMatrix> dropOutMatrices;
    private double learningRate;
    private NeuralNetworkModel neuralNetworkModel;
    private final Supplier<List<DoubleMatrix>> dropOutMatricesSupplier;
    private final IGradientDescentProcessProvider gradientDescentProvider;

    public GradientDescentWithDropOutRegularization(double learningRate, NeuralNetworkModel neuralNetworkModel, Supplier<List<DoubleMatrix>> dropOutMatricesSupplier) {
        this.learningRate = learningRate;
        this.neuralNetworkModel = neuralNetworkModel.clone();
        this.dropOutMatricesSupplier = dropOutMatricesSupplier;
        gradientDescentProvider = new GradientDescentWithDropOutRegularizationProcessProvider(dropOutMatricesSupplier);
    }

    @Override
    public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
        dropOutMatrices = dropOutMatricesSupplier.get();
        DoubleMatrix dropOutOutput = y.mulColumnVector(dropOutMatrices.get(dropOutMatrices.size() - 1));
        GradientLayerProvider gradientLayerProvider = gradientDescentProvider.getForwardComputationLauncher().apply(new ForwardComputationContainer(inputMatrix, neuralNetworkModel));
        List<GradientDescentCorrection> gradientDescentCorrections = gradientDescentProvider.getBackwardComputationLauncher()
                .apply(new BackwardComputationContainer(gradientLayerProvider, dropOutOutput, gradientDescentProvider.getFirstErrorComputationLauncher(), gradientDescentProvider.getErrorComputationLauncher()));
        return gradientDescentProvider.getGradientDescentCorrectionsLauncher().apply(new GradientDescentCorrectionsContainer(neuralNetworkModel, gradientDescentCorrections, y.getColumns(), learningRate));
    }

}
