package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider;

import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.cost.CostComputer;
import com.fgiannesini.neuralnetwork.cost.CostComputerBuilder;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.DataContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.GradientDescentCorrection;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container.GradientDescentWithDerivationContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container.GradientDescentWithDerivationCorrectionsContainer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.container.GradientDescentWithDerivationCostComputerContainer;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class GradientDescentWithDerivationProcessProvider implements IGradientDescentWithDerivationProcessProvider {

    private final double step = 0.0001;

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return Function.identity();
    }

    @Override
    public Function<GradientDescentWithDerivationContainer, List<GradientDescentCorrection>> getGradientWithDerivationLauncher() {
        return container -> {
            LayerTypeData output = container.getY();
            LayerTypeData input = container.getInput();
            List<Layer> layers = container.getNeuralNetworkModel().getLayers();
            List<GradientDescentCorrection> corrections = new ArrayList<>();

            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                List<DoubleMatrix> parametersMatrices = layers.get(layerIndex).getParametersMatrix();

                GradientDescentCorrection gradientDescentCorrection = new GradientDescentCorrection();
                for (int parameterIndex = 0; parameterIndex < parametersMatrices.size(); parameterIndex++) {
                    DoubleMatrix correctedMatrix = DoubleMatrix.zeros(parametersMatrices.get(parameterIndex).getRows(), parametersMatrices.get(parameterIndex).getColumns());

                    for (int elementIndex = 0; elementIndex < correctedMatrix.length; elementIndex++) {
                        NeuralNetworkModel modifiedNeuralNetworkModel = container.getNeuralNetworkModel().clone();
                        CostComputer costComputer = container.getCostComputerProcessLauncher().apply(new GradientDescentWithDerivationCostComputerContainer(modifiedNeuralNetworkModel, container.getCostType()));

                        DoubleMatrix modifiedMatrix = modifiedNeuralNetworkModel.getLayers().get(layerIndex).getParametersMatrix().get(parameterIndex);

                        modifiedMatrix.put(elementIndex, modifiedMatrix.get(elementIndex) + step);
                        double superiorStepCost = costComputer.compute(input, output);
                        modifiedMatrix.put(elementIndex, modifiedMatrix.get(elementIndex) - step);

                        modifiedMatrix.put(elementIndex, modifiedMatrix.get(elementIndex) - step);
                        double inferiorStepCost = costComputer.compute(input, output);
                        modifiedMatrix.put(elementIndex, modifiedMatrix.get(elementIndex) + step);

                        double correction = (superiorStepCost - inferiorStepCost) / (2 * step);
                        correctedMatrix.put(elementIndex, correction);
                    }

                    gradientDescentCorrection.addCorrectionResult(correctedMatrix);
                }
                corrections.add(gradientDescentCorrection);

            }

            return corrections;
        };
    }

    @Override
    public Function<GradientDescentWithDerivationCostComputerContainer, CostComputer> getCostComputerBuildingLauncher() {
        return container -> CostComputerBuilder.init()
                .withNeuralNetworkModel(container.getNeuralNetworkModel())
                .withType(container.getCostType())
                .build();
    }

    @Override
    public Function<GradientDescentWithDerivationCorrectionsContainer, GradientDescentWithDerivationCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return container -> {
            NeuralNetworkModel correctedNeuralNetworkModel = container.getCorrectedNeuralNetworkModel();
            List<Layer> layers = correctedNeuralNetworkModel.getLayers();
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                GradientDescentCorrection gradientDescentCorrection = container.getGradientDescentCorrections().get(layerIndex);
                List<DoubleMatrix> parameterMatrices = layers.get(layerIndex).getParametersMatrix();
                for (int parameterIndex = 0; parameterIndex < parameterMatrices.size(); parameterIndex++) {
                    parameterMatrices.get(parameterIndex).subi(gradientDescentCorrection.getCorrectionResults().get(parameterIndex).mul(container.getLearningRate()));
                }
            }
            return new GradientDescentWithDerivationCorrectionsContainer(correctedNeuralNetworkModel, container.getGradientDescentCorrections(), container.getInputCount(), container.getLearningRate());
        };
    }
}
