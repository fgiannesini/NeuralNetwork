package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout.DropOutUtils;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout.GradientDescentWithDerivationAndDropOutRegularizationProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout.GradientDescentWithDropOutRegularizationProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2.GradientDescentWithDerivationAndL2RegularizationProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2.GradientDescentWithL2RegularizationProcessProvider;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

public class LearningAlgorithmBuilder {

    private NeuralNetworkModel neuralNetworkModel;

    private LearningAlgorithmType learningAlgorithmType;
    private double learningRate;
    private CostType costType;
    private Double l2RegularizationCoeff;
    private double[] dropOutRegularizationCoeffs;
    private Double momentumCoeff;
    private Double rmsStopCoeff;

    private LearningAlgorithmBuilder() {
        learningAlgorithmType = LearningAlgorithmType.GRADIENT_DESCENT;
        learningRate = 0.01;
        costType = CostType.LINEAR_REGRESSION;
        momentumCoeff = 0.9;
        rmsStopCoeff = 0.9;
    }

    public static LearningAlgorithmBuilder init() {
        return new LearningAlgorithmBuilder();
    }

    public LearningAlgorithmBuilder withModel(NeuralNetworkModel neuralNetworkModel) {
        this.neuralNetworkModel = neuralNetworkModel;
        return this;
    }

    public LearningAlgorithmBuilder withAlgorithmType(LearningAlgorithmType learningAlgorithmType) {
        this.learningAlgorithmType = learningAlgorithmType;
        return this;
    }

    public LearningAlgorithmBuilder withLearningRate(double learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    public LearningAlgorithmBuilder withCostType(CostType costType) {
        this.costType = costType;
        return this;
    }

    public LearningAlgorithmBuilder withL2Regularization(Double l2RegularizationCoeff) {
        this.l2RegularizationCoeff = l2RegularizationCoeff;
        return this;
    }

    public LearningAlgorithmBuilder withDropOutRegularitation(double... dropOutRegularizationCoeffs) {
        this.dropOutRegularizationCoeffs = dropOutRegularizationCoeffs;
        return this;
    }

    public LearningAlgorithmBuilder withMomentumCoeff(Double momentumCoeff) {
        this.momentumCoeff = momentumCoeff;
        return this;
    }

    public LearningAlgorithmBuilder withRmsStopCoeff(Double rmsStopCoeff) {
        this.rmsStopCoeff = rmsStopCoeff;
        return this;
    }

    public LearningAlgorithm build() {
        checkInputs();
        LearningAlgorithm learningAlgorithm;
        switch (learningAlgorithmType) {
            case GRADIENT_DESCENT:
                IGradientDescentProcessProvider processProvider = applyGradientDescentRegularization(new GradientDescentProcessProvider());
                learningAlgorithm = new GradientDescent(neuralNetworkModel, learningRate, processProvider);
                break;
            case GRADIENT_DESCENT_MOMENTUM:
                GradientDescentWithMomentumProcessProvider withMomentumProcessProvider = new GradientDescentWithMomentumProcessProvider(momentumCoeff);
                processProvider = applyGradientDescentRegularization(withMomentumProcessProvider);
                learningAlgorithm = new GradientDescent(neuralNetworkModel, learningRate, processProvider);
                break;
            case GRADIENT_DESCENT_RMS_STOP:
                processProvider = new GradientDescentWithRmsStopProcessProvider(rmsStopCoeff);
                processProvider = applyGradientDescentRegularization(processProvider);
                learningAlgorithm = new GradientDescent(neuralNetworkModel, learningRate, processProvider);
                break;
            case GRADIENT_DESCENT_DERIVATION:
                IGradientDescentWithDerivationProcessProvider withDerivationProcessProvider = new GradientDescentWithDerivationProcessProvider();
                withDerivationProcessProvider = applyGradientDescentWithDerivationRegularization(withDerivationProcessProvider);
                learningAlgorithm = new GradientDescentWithDerivation(neuralNetworkModel, costType, learningRate, withDerivationProcessProvider);
                break;
            case GRADIENT_DESCENT_DERIVATION_MOMENTUM:
                GradientDescentWithDerivationAndMomentumProcessProvider withDerivationAndMomentumProcessProvider = new GradientDescentWithDerivationAndMomentumProcessProvider(momentumCoeff);
                withDerivationProcessProvider = applyGradientDescentWithDerivationRegularization(withDerivationAndMomentumProcessProvider);
                learningAlgorithm = new GradientDescentWithDerivation(neuralNetworkModel, costType, learningRate, withDerivationProcessProvider);
                break;
            case GRADIENT_DESCENT_DERIVATION_RMS_STOP:
                GradientDescentWithDerivationAndRmsStopProcessProvider withDerivationAndRmsStopProcessProvider = new GradientDescentWithDerivationAndRmsStopProcessProvider(rmsStopCoeff);
                withDerivationProcessProvider = applyGradientDescentWithDerivationRegularization(withDerivationAndRmsStopProcessProvider);
                learningAlgorithm = new GradientDescentWithDerivation(neuralNetworkModel, costType, learningRate, withDerivationProcessProvider);
                break;
            default:
                throw new IllegalArgumentException(learningAlgorithmType + " instantiation is not implemented");
        }
        return learningAlgorithm;
    }

    private IGradientDescentWithDerivationProcessProvider applyGradientDescentWithDerivationRegularization(IGradientDescentWithDerivationProcessProvider withDerivationProcessProvider) {
        if (dropOutRegularizationCoeffs != null) {
            Supplier<List<DoubleMatrix>> dropOutMatricesSupplier = () -> DropOutUtils.init().getDropOutMatrix(dropOutRegularizationCoeffs, neuralNetworkModel.getLayers());
            withDerivationProcessProvider = new GradientDescentWithDerivationAndDropOutRegularizationProcessProvider(dropOutMatricesSupplier, withDerivationProcessProvider);
        } else if (l2RegularizationCoeff != null) {
            withDerivationProcessProvider = new GradientDescentWithDerivationAndL2RegularizationProcessProvider(l2RegularizationCoeff, withDerivationProcessProvider);
        }
        return withDerivationProcessProvider;
    }

    private IGradientDescentProcessProvider applyGradientDescentRegularization(IGradientDescentProcessProvider processProvider) {
        if (dropOutRegularizationCoeffs != null) {
            Supplier<List<DoubleMatrix>> dropOutMatricesSupplier = () -> DropOutUtils.init().getDropOutMatrix(dropOutRegularizationCoeffs, neuralNetworkModel.getLayers());
            processProvider = new GradientDescentWithDropOutRegularizationProcessProvider(dropOutMatricesSupplier, processProvider);
        } else if (l2RegularizationCoeff != null) {
            processProvider = new GradientDescentWithL2RegularizationProcessProvider(l2RegularizationCoeff, neuralNetworkModel, processProvider);
        }
        return processProvider;
    }

    private void checkInputs() {
        if (neuralNetworkModel == null) {
            throw new IllegalArgumentException("NeuralNetworkModel missing");
        }
        if (dropOutRegularizationCoeffs != null && l2RegularizationCoeff != null) {
            throw new IllegalArgumentException("You can't use several regularization methods");
        }
        if (dropOutRegularizationCoeffs != null && dropOutRegularizationCoeffs.length != neuralNetworkModel.getLayers().size() + 1) {
            throw new IllegalArgumentException("Drop out Regularization needs " + (neuralNetworkModel.getLayers().size() + 1) + " parameters");
        }
        if (dropOutRegularizationCoeffs != null && Arrays.stream(dropOutRegularizationCoeffs).anyMatch(d -> d > 1 || d < 0)) {
            throw new IllegalArgumentException("Drop out Regularization coeffs should be between 0 and 1");
        }
    }

}
