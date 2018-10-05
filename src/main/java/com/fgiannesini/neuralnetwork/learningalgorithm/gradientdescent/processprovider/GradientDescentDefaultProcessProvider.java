package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IIntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProviderBuilder;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;
import java.util.function.Function;

public class GradientDescentDefaultProcessProvider implements IGradientDescentProcessProvider {

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
            GradientDescentErrorComputationVisitor errorComputationVisitor = new GradientDescentErrorComputationVisitor(container.getProvider());
            container.getPreviousError().accept(errorComputationVisitor);
            return new ErrorComputationContainer(container.getProvider(), errorComputationVisitor.getErrorData());
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
