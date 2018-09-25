package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IIntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientBatchNormLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProviderBuilder;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.function.Function;

public class GradientDescentBatchNormProcessProvider implements IGradientDescentProcessProvider {

    @Override
    public IGradientDescentProcessProvider getPreviousProcessProvider() {
        throw new RuntimeException("Should not be called");
    }

    @Override
    public Function<GradientDescentCorrectionsContainer, GradientDescentCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return container -> {
            NeuralNetworkModel correctedNeuralNetworkModel = container.getCorrectedNeuralNetworkModel();
            List<Layer> layers = correctedNeuralNetworkModel.getLayers();
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                GradientDescentCorrection gradientDescentCorrection = container.getGradientDescentCorrections().get(layerIndex);
                Layer layer = layers.get(layerIndex);
                List<DoubleMatrix> parametersMatrices = layer.getParametersMatrix();
                for (int parameterIndex = 0; parameterIndex < parametersMatrices.size(); parameterIndex++) {
                    parametersMatrices.get(parameterIndex).subi(gradientDescentCorrection.getCorrectionResults().get(parameterIndex).mul(container.getLearningRate()));
                }
            }
            return new GradientDescentCorrectionsContainer(correctedNeuralNetworkModel, container.getGradientDescentCorrections(), container.getInputCount(), container.getLearningRate());
        };
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return container -> {
            //dZ1 = W2t * dZ2 .* g1'(A1)
            GradientBatchNormLayerProvider provider = (GradientBatchNormLayerProvider) container.getProvider();
            BatchNormData batchNormData = (BatchNormData) container.getPreviousError();
            DoubleMatrix previousError = batchNormData.getInput();
            DoubleMatrix error = provider.getPreviousWeightMatrix().transpose()
                    .mmul(previousError)
                    .muli(provider.getCurrentActivationFunction().derivate(provider.getCurrentResult()));
            return new ErrorComputationContainer(provider, new BatchNormData(error, batchNormData.getMeanDeviationProvider()));
        };
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return container -> {
            throw new RuntimeException("Should use a regression type process provider");
        };
    }

    @Override
    public Function<ForwardComputationContainer, List<GradientLayerProvider>> getForwardComputationLauncher() {
        return container -> {
            NeuralNetworkModel neuralNetworkModel = container.getNeuralNetworkModel();
            IIntermediateOutputComputer intermediateOutputComputer = OutputComputerBuilder.init()
                    .withModel(neuralNetworkModel)
                    .buildIntermediateOutputComputer();
            LayerTypeData inputData = container.getInput();
            List<IntermediateOutputResult> intermediateResults = intermediateOutputComputer.compute(inputData);
            return GradientLayerProviderBuilder.init()
                    .withModel(neuralNetworkModel)
                    .withIntermediateResults(intermediateResults)
                    .build();
        };
    }

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return Function.identity();
    }
}
