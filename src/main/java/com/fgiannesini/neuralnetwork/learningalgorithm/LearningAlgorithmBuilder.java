package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.GradientDescent;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout.DropOutUtils;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout.GradientDescentWithDerivationAndDropOutRegularization;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout.GradientDescentWithDropOutRegularization;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2.GradientDescentWithDerivationAndL2Regularization;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2.GradientDescentWithL2Regularization;
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

    private LearningAlgorithmBuilder() {
        learningAlgorithmType = LearningAlgorithmType.GRADIENT_DESCENT;
        learningRate = 0.01;
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

    public LearningAlgorithm build() {
        checkInputs();
        LearningAlgorithm learningAlgorithm;
        switch (learningAlgorithmType) {
            case GRADIENT_DESCENT:
                GradientDescent gradientDescent = new GradientDescent(neuralNetworkModel, learningRate);
                if (dropOutRegularizationCoeffs != null) {
                    Supplier<List<DoubleMatrix>> dropOutMatricesSupplier = () -> DropOutUtils.init().getDropOutMatrix(dropOutRegularizationCoeffs, neuralNetworkModel.getLayers());
                    learningAlgorithm = new GradientDescentWithDropOutRegularization(gradientDescent, learningRate, neuralNetworkModel, dropOutMatricesSupplier);
                } else if (l2RegularizationCoeff != null) {
                    learningAlgorithm = new GradientDescentWithL2Regularization(neuralNetworkModel, learningRate, l2RegularizationCoeff);
                } else {
                    learningAlgorithm = gradientDescent;
                }
                break;
            case GRADIENT_DESCENT_DERIVATION:
                if (dropOutRegularizationCoeffs != null) {
                    Supplier<List<DoubleMatrix>> dropOutMatricesSupplier = () -> DropOutUtils.init().getDropOutMatrix(dropOutRegularizationCoeffs, neuralNetworkModel.getLayers());
                    learningAlgorithm = new GradientDescentWithDerivationAndDropOutRegularization(neuralNetworkModel, costType, learningRate, dropOutMatricesSupplier);
                } else if (l2RegularizationCoeff != null) {
                    learningAlgorithm = new GradientDescentWithDerivationAndL2Regularization(neuralNetworkModel, costType, learningRate, l2RegularizationCoeff);
                } else {
                    learningAlgorithm = new GradientDescentWithDerivation(neuralNetworkModel, costType, learningRate);
                }
                break;
            default:
                throw new IllegalArgumentException(learningAlgorithmType + " instantiation is not implemented");
        }
        return learningAlgorithm;
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
