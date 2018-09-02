package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IIntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientBatchNormLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.*;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

public class GradientDescentBatchNormProcessProvider implements IGradientDescentProcessProvider<BatchNormLayer> {

    public GradientDescentBatchNormProcessProvider() {
    }

    @Override
    public Function<GradientDescentCorrectionsContainer<BatchNormLayer>, GradientDescentCorrectionsContainer<BatchNormLayer>> getGradientDescentCorrectionsLauncher() {
        return container -> {
            NeuralNetworkModel<BatchNormLayer> correctedNeuralNetworkModel = container.getCorrectedNeuralNetworkModel();
            List<BatchNormLayer> layers = correctedNeuralNetworkModel.getLayers();
            for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
                GradientDescentCorrection gradientDescentCorrection = container.getGradientDescentCorrections().get(layerIndex);
                List<DoubleMatrix> parametersMatrices = layers.get(layerIndex).getParametersMatrix();
                for (int parameterIndex = 0; parameterIndex < parametersMatrices.size(); parameterIndex++) {
                    parametersMatrices.get(parameterIndex).subi(gradientDescentCorrection.getCorrectionResults().get(parameterIndex).mul(container.getLearningRate()));
                }
            }
            return new GradientDescentCorrectionsContainer<>(correctedNeuralNetworkModel, container.getGradientDescentCorrections(), container.getInputCount(), container.getLearningRate());
        };
    }

    @Override
    public Function<BackwardComputationContainer, List<GradientDescentCorrection>> getBackwardComputationLauncher() {
        return container -> {
            List<GradientDescentCorrection> gradientDescentCorrections = new ArrayList<>();
//            int inputCount = container.getY().getColumns();
//
//            GradientLayerProvider gradientLayerProvider = container.getProvider();
//            DoubleMatrix dz = container.getFirstErrorComputationLauncher()
//                    .apply(new ErrorComputationContainer(gradientLayerProvider, container.getY()))
//                    .getPreviousError();
//            DoubleMatrix weightCorrection = computeWeightCorrection(gradientLayerProvider.getPreviousResult(), dz, inputCount);
//            DoubleMatrix biasCorrection = computeBiasCorrection(dz, inputCount);
//
//            gradientDescentCorrections.add(new GradientDescentCorrection(weightCorrection, biasCorrection));
//
//            for (gradientLayerProvider.nextLayer(); gradientLayerProvider.hasNextLayer(); gradientLayerProvider.nextLayer()) {
//                dz = container.getErrorComputationLauncher()
//                        .apply(new ErrorComputationContainer(gradientLayerProvider, dz))
//                        .getPreviousError();
//                weightCorrection = computeWeightCorrection(gradientLayerProvider.getPreviousResult(), dz, inputCount);
//                biasCorrection = computeBiasCorrection(dz, inputCount);
//                gradientDescentCorrections.add(new GradientDescentCorrection(weightCorrection, biasCorrection));
//            }
//

            int inputCount = container.getY().getColumns();

            GradientBatchNormLayerProvider gradientLayerProvider = (GradientBatchNormLayerProvider) container.getProvider();
            DoubleMatrix dz = container.getFirstErrorComputationLauncher()
                    .apply(new ErrorComputationContainer(gradientLayerProvider, container.getY()))
                    .getPreviousError();

//  #step9
//            dbeta = np.sum(dout, axis=0)
            DoubleMatrix dBeta = dz.columnSums();
//            dgammax = dout #not necessary, but more understandable
            DoubleMatrix dGammaX = dz;

//  #step8
//            dgamma = np.sum(dgammax*xhat, axis=0)
            DoubleMatrix dGamma = dGammaX.mul(gradientLayerProvider.getCurrentResult());
//            dxhat = dgammax * gamma
            DoubleMatrix dXhat = dGammaX.mul(gradientLayerProvider.getGammaMatrix());

//  #step7
//                    divar = np.sum(dxhat*xmu, axis=0)
//            dxmu1 = dxhat * ivar
//
//  #step6
//                    dsqrtvar = -1. /(sqrtvar**2) * divar
//
//  #step5
//                    dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar
//
//  #step4
//                    dsq = 1. /N * np.ones((N,D)) * dvar
//
//  #step3
//                    dxmu2 = 2 * xmu * dsq
//
//  #step2
//                    dx1 = (dxmu1 + dxmu2)
//            dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
//
//  #step1
//                    dx2 = 1. /N * np.ones((N,D)) * dmu
//
//  #step0
//                    dx = dx1 + dx2
//
//            return dx, dgamma, dbeta

            Collections.reverse(gradientDescentCorrections);
            return gradientDescentCorrections;
        };

    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return container -> {
            throw new RuntimeException("Should use a regression type process provider");
        };
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return container -> {
            throw new RuntimeException("Should use a regression type process provider");
        };
    }

    @Override
    public Function<ForwardComputationContainer<BatchNormLayer>, GradientLayerProvider<BatchNormLayer>> getForwardComputationLauncher() {
        return container -> {
            NeuralNetworkModel<BatchNormLayer> neuralNetworkModel = container.getNeuralNetworkModel();
            List<BatchNormLayer> layers = neuralNetworkModel.getLayers();
            IIntermediateOutputComputer intermediateOutputComputer = OutputComputerBuilder.init()
                    .withModel(neuralNetworkModel)
                    .buildIntermediateOutputComputer();
            List<DoubleMatrix> intermediateResults = intermediateOutputComputer.compute(container.getInputMatrix());
            return new GradientBatchNormLayerProvider(layers, intermediateResults);
        };
    }

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return Function.identity();
    }
}
