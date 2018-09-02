package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IIntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientBatchNormLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.layerdataprovider.GradientLayerProviderBuilder;
import com.fgiannesini.neuralnetwork.model.BatchNormLayer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;

public class GradientDescentBatchNormProcessProvider implements IGradientDescentProcessProvider<BatchNormLayer> {

    private double epsilon;

    public GradientDescentBatchNormProcessProvider() {
        epsilon = Math.pow(10, -8);
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
            int inputCount = container.getY().getColumns();

            GradientBatchNormLayerProvider gradientLayerProvider = (GradientBatchNormLayerProvider) container.getProvider();
            DoubleMatrix dz = container.getFirstErrorComputationLauncher()
                    .apply(new ErrorComputationContainer(gradientLayerProvider, container.getY()))
                    .getPreviousError();
            BatchNormBackwardReturn batchNormBackwardReturn = getBatchNormBackwardReturn(inputCount, gradientLayerProvider, dz);

            gradientDescentCorrections.add(batchNormBackwardReturn.getCorrections());

            for (gradientLayerProvider.nextLayer(); gradientLayerProvider.hasNextLayer(); gradientLayerProvider.nextLayer()) {
                dz = container.getErrorComputationLauncher()
                        .apply(new ErrorComputationContainer(gradientLayerProvider, batchNormBackwardReturn.getNextError()))
                        .getPreviousError();
                batchNormBackwardReturn = getBatchNormBackwardReturn(inputCount, gradientLayerProvider, dz);
                gradientDescentCorrections.add(batchNormBackwardReturn.getCorrections());
            }

            Collections.reverse(gradientDescentCorrections);
            return gradientDescentCorrections;
        };

    }

    public BatchNormBackwardReturn getBatchNormBackwardReturn(int inputCount, GradientBatchNormLayerProvider gradientLayerProvider, DoubleMatrix dz) {
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
        DoubleMatrix diVar = dXhat.mul(gradientLayerProvider.getMean());
//            dxmu1 = dxhat * ivar
        DoubleMatrix dXmu1 = dXhat.div(gradientLayerProvider.getStandardDeviation());
//
//  #step6
//                    dsqrtvar = -1. /(sqrtvar**2) * divar
        DoubleMatrix dSqrtVar = diVar.mul(MatrixFunctions.pow(gradientLayerProvider.getStandardDeviation(), 2).muli(-1));
//
//  #step5
//                    dvar = 0.5 * 1. /np.sqrt(var+eps) * dsqrtvar

        DoubleMatrix dVar = dSqrtVar.div(MatrixFunctions.sqrt(gradientLayerProvider.getStandardDeviation().add(epsilon)));
//
//  #step4
//                    dsq = 1. /N * np.ones((N,D)) * dvar
        DoubleMatrix dsq = dVar.div(DoubleMatrix.ones(gradientLayerProvider.getInputSize(), gradientLayerProvider.getOutputSize()));
//
//  #step3
//                    dxmu2 = 2 * xmu * dsq
        DoubleMatrix dXmu2 = gradientLayerProvider.getMean().mul(dsq).mul(2);
//
//  #step2
//                    dx1 = (dxmu1 + dxmu2)
        DoubleMatrix dX1 = dXmu1.add(dXmu2);
//            dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
        DoubleMatrix dMu = dXmu1.add(dXmu2).columnSums().muli(-1);
//
//  #step1
//                    dx2 = 1. /N * np.ones((N,D)) * dmu
        DoubleMatrix dx2 = dMu.div(DoubleMatrix.ones(gradientLayerProvider.getInputSize(), gradientLayerProvider.getOutputSize()));
//
//  #step0
//                    dx = dx1 + dx2
        DoubleMatrix dx = dX1.add(dx2);
        DoubleMatrix weightCorrection = computeWeightCorrection(gradientLayerProvider.getPreviousResult(), dx, inputCount);
        return new BatchNormBackwardReturn(weightCorrection, dGamma, dBeta, dx);
    }

    private DoubleMatrix computeWeightCorrection(DoubleMatrix previousaLayerResult, DoubleMatrix dz, int inputCount) {
        //dW1 = dZ1 * A0t ./m
        return dz
                .mmul(previousaLayerResult.transpose())
                .divi(inputCount);
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return container -> {
            //dZ1 = W2t * dZ2 .* g1'(A1)
            DoubleMatrix error = container.getProvider().getPreviousWeightMatrix().transpose()
                    .mmul(container.getPreviousError())
                    .muli(container.getProvider().getCurrentActivationFunction().derivate(container.getProvider().getCurrentResult()));
            return new ErrorComputationContainer(container.getProvider(), error);
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
            IIntermediateOutputComputer intermediateOutputComputer = OutputComputerBuilder.init()
                    .withModel(neuralNetworkModel)
                    .buildIntermediateOutputComputer();
            List<IntermediateOutputResult> intermediateResults = intermediateOutputComputer.compute(container.getInputMatrix());
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
