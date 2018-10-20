package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.adapter.BackwardDataAdapterVisitor;
import com.fgiannesini.neuralnetwork.learningalgorithm.LearningAlgorithm;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.IGradientDescentProcessProvider;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class GradientDescent implements LearningAlgorithm {
    private final IGradientDescentProcessProvider gradientDescentProcessProvider;
    private double learningRate;
    private NeuralNetworkModel correctedNeuralNetworkModel;

    public GradientDescent(NeuralNetworkModel originalNeuralNetworkModel, IGradientDescentProcessProvider gradientDescentProcessProvider) {
        this.gradientDescentProcessProvider = gradientDescentProcessProvider;
        this.correctedNeuralNetworkModel = originalNeuralNetworkModel.clone();
        this.learningRate = 0.01;
    }

    @Override
    public NeuralNetworkModel learn(LayerTypeData inputData, LayerTypeData outputData) {
        DataContainer dataContainer = new DataContainer(inputData, outputData);
        dataContainer = gradientDescentProcessProvider.getDataProcessLauncher().apply(dataContainer);

        List<GradientLayerProvider> providers = gradientDescentProcessProvider.getForwardComputationLauncher()
                .apply(new ForwardComputationContainer(dataContainer.getInput(), correctedNeuralNetworkModel));

        Collections.reverse(providers);

        List<GradientDescentCorrection> gradientDescentCorrections = new ArrayList<>();

        LayerTypeData firstError = gradientDescentProcessProvider.getFirstErrorComputationLauncher().apply(new ErrorComputationContainer(providers.get(0), dataContainer.getOutput())).getPreviousError();
        LayerTypeCorrectionsVisitor firstLayerTypeCorrectionsVisitor = new LayerTypeCorrectionsVisitor(providers.get(0));
        firstError.accept(firstLayerTypeCorrectionsVisitor);
        gradientDescentCorrections.add(firstLayerTypeCorrectionsVisitor.getCorrection());

        LayerTypeData nextError = firstLayerTypeCorrectionsVisitor.getNextGradientLayerProvider();
        for (int i = 1; i < providers.size(); i++) {
            GradientLayerProvider provider = providers.get(i);
            BackwardDataAdapterVisitor backwardDataAdapterVisitor = new BackwardDataAdapterVisitor(nextError);
            provider.getLayer().accept(backwardDataAdapterVisitor);
            LayerTypeData activatedError = gradientDescentProcessProvider.getErrorComputationLauncher()
                    .apply(new ErrorComputationContainer(provider, backwardDataAdapterVisitor.getData()))
                    .getPreviousError();
            LayerTypeCorrectionsVisitor layerTypeCorrectionsVisitor = new LayerTypeCorrectionsVisitor(provider);
            activatedError.accept(layerTypeCorrectionsVisitor);
            nextError = layerTypeCorrectionsVisitor.getNextGradientLayerProvider();
            GradientDescentCorrection correction = layerTypeCorrectionsVisitor.getCorrection();
            gradientDescentCorrections.add(correction);
        }

        Collections.reverse(gradientDescentCorrections);

        correctedNeuralNetworkModel = gradientDescentProcessProvider.getGradientDescentCorrectionsLauncher()
                .apply(new GradientDescentCorrectionsContainer(this.correctedNeuralNetworkModel, gradientDescentCorrections, dataContainer.getOutput().getInputCount(), learningRate))
                .getCorrectedNeuralNetworkModel();
        return correctedNeuralNetworkModel;
    }

    @Override
    public void updateLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public IGradientDescentProcessProvider getGradientDescentProcessProvider() {
        return gradientDescentProcessProvider;
    }
}
