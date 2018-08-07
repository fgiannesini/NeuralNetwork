package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;

public class GradientDescent implements LearningAlgorithm {
    private final NeuralNetworkModel correctedNeuralNetworkModel;
    private final double learningRate;
    private final IGradientDescentProcessProvider gradientDescentProvider;

    public GradientDescent( NeuralNetworkModel originalNeuralNetworkModel, double learningRate) {
        this.gradientDescentProvider = new GradientDescentProcessProvider();
        this.correctedNeuralNetworkModel = originalNeuralNetworkModel.clone();
        this.learningRate = learningRate;
    }

    @Override
    public NeuralNetworkModel learn(DoubleMatrix inputMatrix, DoubleMatrix y) {
        DataContainer dataContainer = new DataContainer(inputMatrix, y);
        dataContainer = gradientDescentProvider.getDataProcessLauncher().apply(dataContainer);

        GradientLayerProvider provider = gradientDescentProvider.getForwardComputationLauncher()
                .apply(new ForwardComputationContainer(dataContainer.getInput(), correctedNeuralNetworkModel));

        List<GradientDescentCorrection> gradientDescentCorrections = gradientDescentProvider.getBackwardComputationLauncher()
                .apply(new BackwardComputationContainer(provider, dataContainer.getOutput(), gradientDescentProvider.getFirstErrorComputationLauncher(), gradientDescentProvider.getErrorComputationLauncher()));

        return gradientDescentProvider.getGradientDescentCorrectionsLauncher()
                .apply(new GradientDescentCorrectionsContainer(correctedNeuralNetworkModel, gradientDescentCorrections, dataContainer.getOutput().getColumns(), learningRate))
                .getCorrectedNeuralNetworkModel();
    }

}
