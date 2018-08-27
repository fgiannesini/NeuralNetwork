package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientLayerProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.container.*;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class GradientDescentBatchNormProcessProvider implements IGradientDescentProcessProvider {
    private IGradientDescentProcessProvider processProvider;

    public GradientDescentBatchNormProcessProvider(IGradientDescentProcessProvider processProvider) {
        this.processProvider = processProvider;
    }

    @Override
    public Function<GradientDescentCorrectionsContainer, GradientDescentCorrectionsContainer> getGradientDescentCorrectionsLauncher() {
        return processProvider.getGradientDescentCorrectionsLauncher();
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
//            Collections.reverse(gradientDescentCorrections);

            return gradientDescentCorrections;
        };
//            #unfold the variables stored in cache
//            xhat,gamma,xmu,ivar,sqrtvar,var,eps = cache
//
//  #get the dimensions of the input/output
//                    N,D = dout.shape
//
//  #step9
//                    dbeta = np.sum(dout, axis=0)
//            dgammax = dout #not necessary, but more understandable
//
//  #step8
//                    dgamma = np.sum(dgammax*xhat, axis=0)
//            dxhat = dgammax * gamma
//
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
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getErrorComputationLauncher() {
        return processProvider.getErrorComputationLauncher();
    }

    @Override
    public Function<ErrorComputationContainer, ErrorComputationContainer> getFirstErrorComputationLauncher() {
        return processProvider.getFirstErrorComputationLauncher();
    }

    @Override
    public Function<ForwardComputationContainer, GradientLayerProvider> getForwardComputationLauncher() {
        return processProvider.getForwardComputationLauncher();
    }

    @Override
    public Function<DataContainer, DataContainer> getDataProcessLauncher() {
        return processProvider.getDataProcessLauncher();
    }
}
