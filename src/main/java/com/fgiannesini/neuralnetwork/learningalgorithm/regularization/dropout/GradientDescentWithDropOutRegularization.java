package com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout;

import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.*;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.function.Supplier;

public class GradientDescentWithDropOutRegularization implements LearningAlgorithm {

    private final double learningRate;
    private final NeuralNetworkModel neuralNetworkModel;
    private final IGradientDescentProcessProvider gradientDescentProvider;

    public GradientDescentWithDropOutRegularization(double learningRate, NeuralNetworkModel neuralNetworkModel, Supplier<List<DoubleMatrix>> dropOutMatricesSupplier) {
        this.learningRate = learningRate;
        this.neuralNetworkModel = neuralNetworkModel.clone();
        gradientDescentProvider = new GradientDescentWithDropOutRegularizationProcessProvider(dropOutMatricesSupplier);
    }

    @Override
    public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
        DataContainer dataContainer =  gradientDescentProvider.getDataProcessLauncher().apply(new DataContainer(inputMatrix, y));

        GradientLayerProvider gradientLayerProvider = gradientDescentProvider.getForwardComputationLauncher()
                .apply(new ForwardComputationContainer(inputMatrix, neuralNetworkModel));

        List<GradientDescentCorrection> gradientDescentCorrections = gradientDescentProvider.getBackwardComputationLauncher()
                .apply(new BackwardComputationContainer(gradientLayerProvider, dataContainer.getOutput(), gradientDescentProvider.getFirstErrorComputationLauncher(), gradientDescentProvider.getErrorComputationLauncher()));

        return gradientDescentProvider.getGradientDescentCorrectionsLauncher()
                .apply(new GradientDescentCorrectionsContainer(neuralNetworkModel, gradientDescentCorrections, y.getColumns(), learningRate))
                .getCorrectedNeuralNetworkModel();
    }

}
