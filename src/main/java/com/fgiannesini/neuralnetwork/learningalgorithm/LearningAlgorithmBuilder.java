package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.processprovider.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.processprovider.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout.DropOutUtils;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout.GradientDescentWithDerivationAndDropOutRegularizationProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout.GradientDescentWithDropOutRegularizationProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2.GradientDescentWithDerivationAndL2RegularizationProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2.GradientDescentWithL2RegularizationProcessProvider;
import com.fgiannesini.neuralnetwork.model.LayerType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.Arrays;
import java.util.List;
import java.util.function.Supplier;

public class LearningAlgorithmBuilder {

    private NeuralNetworkModel neuralNetworkModel;

    private LearningAlgorithmType learningAlgorithmType;
    private CostType costType;
    private Double l2RegularizationCoeff;
    private double[] dropOutRegularizationCoeffs;
    private Double momentumCoeff;
    private Double rmsStopCoeff;

    private LearningAlgorithmBuilder() {
        learningAlgorithmType = LearningAlgorithmType.GRADIENT_DESCENT;
        costType = CostType.LINEAR_REGRESSION;
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
                IGradientDescentProcessProvider processProvider = new GradientDescentDefaultProcessProvider();
                if (neuralNetworkModel.getLayerType().equals(LayerType.BATCH_NORM)) {
                    processProvider = new GradientDescentBatchNormProcessProvider(processProvider);
                }
                switch (costType) {
                    case LINEAR_REGRESSION:
                        processProvider = new GradientDescentOnLinearRegressionProcessProvider(processProvider);
                        break;
                    case LOGISTIC_REGRESSION:
                        processProvider = new GradientDescentOnLogisticRegressionProcessProvider(processProvider);
                        break;
                    case SOFT_MAX_REGRESSION:
                        processProvider = new GradientDescentOnSoftMaxRegressionProcessProvider(processProvider);
                        break;
                    default:
                        throw new IllegalArgumentException("a cost type should be set");
                }
                processProvider = applyGradientDescentOptimisation(processProvider);
                processProvider = applyGradientDescentRegularization(processProvider);
                learningAlgorithm = new GradientDescent(neuralNetworkModel, processProvider);
                break;

            case GRADIENT_DESCENT_DERIVATION:
                IGradientDescentWithDerivationProcessProvider linearRegressionDerivationProcessProvider = new GradientDescentWithDerivationProcessProvider();
                linearRegressionDerivationProcessProvider = applyGradientDescentWithDerivationOptimisation(linearRegressionDerivationProcessProvider);
                linearRegressionDerivationProcessProvider = applyGradientDescentWithDerivationRegularization(linearRegressionDerivationProcessProvider);
                learningAlgorithm = new GradientDescentWithDerivation(neuralNetworkModel, costType, linearRegressionDerivationProcessProvider);
                break;
            default:
                throw new IllegalArgumentException(learningAlgorithmType + " instantiation is not implemented");
        }
        return learningAlgorithm;
    }

    private IGradientDescentWithDerivationProcessProvider applyGradientDescentWithDerivationOptimisation(IGradientDescentWithDerivationProcessProvider processProvider) {
        if (momentumCoeff != null && rmsStopCoeff != null) {
            processProvider = new GradientDescentWithDerivationAndAdamOptimisationProcessProvider(processProvider, momentumCoeff, rmsStopCoeff);
        } else if (momentumCoeff != null) {
            processProvider = new GradientDescentWithDerivationAndMomentumProcessProvider(processProvider, momentumCoeff);
        } else if (rmsStopCoeff != null) {
            processProvider = new GradientDescentWithDerivationAndRmsStopProcessProvider(processProvider, rmsStopCoeff);
        }
        return processProvider;
    }

    private IGradientDescentProcessProvider applyGradientDescentOptimisation(IGradientDescentProcessProvider processProvider) {
        if (momentumCoeff != null && rmsStopCoeff != null) {
            processProvider = new GradientDescentWithAdamOptimisationProcessProvider(processProvider, momentumCoeff, rmsStopCoeff);
        } else if (momentumCoeff != null) {
            processProvider = new GradientDescentWithMomentumProcessProvider(processProvider, momentumCoeff);
        } else if (rmsStopCoeff != null) {
            processProvider = new GradientDescentWithRmsStopProcessProvider(processProvider, rmsStopCoeff);
        }
        return processProvider;
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
