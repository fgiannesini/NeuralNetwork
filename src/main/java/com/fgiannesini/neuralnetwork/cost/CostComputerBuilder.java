package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.computer.OutputComputerBuilder;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;

public class CostComputerBuilder {

    private NeuralNetworkModel neuralNetworkModel;
    private CostType costType;
    private Double l2RegularizationCoeff;
    private List<DoubleMatrix> dropOutCoeffs;

    private CostComputerBuilder() {
        costType = CostType.LINEAR_REGRESSION;
    }

    public static CostComputerBuilder init() {
        return new CostComputerBuilder();
    }

    public CostComputerBuilder withNeuralNetworkModel(NeuralNetworkModel neuralNetworkModel) {
        this.neuralNetworkModel = neuralNetworkModel;
        return this;
    }

    public CostComputerBuilder withType(CostType costType) {
        this.costType = costType;
        return this;
    }

    public CostComputerBuilder withDropOutRegularization(List<DoubleMatrix> dropOutCoeffs) {
        this.dropOutCoeffs = dropOutCoeffs;
        return this;
    }

    public CostComputerBuilder withL2Regularization(double l2RegularizationCoeff) {
        this.l2RegularizationCoeff = l2RegularizationCoeff;
        return this;
    }

    public CostComputer build() {
        checkInputs();
        IFinalOutputComputer outputComputer = buildFinalOutputComputer();
        CostComputer costComputer;
        switch (costType) {
            case LOGISTIC_REGRESSION:
                costComputer = new LogisticRegressionCostComputer(outputComputer);
                break;
            case LINEAR_REGRESSION:
                costComputer = new LinearRegressionCostComputer(outputComputer);
                break;
            default:
                throw new IllegalArgumentException(costType + " instantiation is not implemented");
        }

        if (l2RegularizationCoeff != null) {
            costComputer = new CostComputerWithL2Regularization(neuralNetworkModel, costComputer, l2RegularizationCoeff);
        }

        return costComputer;
    }

    private void checkInputs() {
        if (neuralNetworkModel == null) {
            throw new IllegalArgumentException("NeuralNetworkModel missing");
        }
        if (l2RegularizationCoeff != null && dropOutCoeffs != null) {
            throw new IllegalArgumentException("You can't use many regularization methods");
        }
    }

    private IFinalOutputComputer buildFinalOutputComputer() {
        OutputComputerBuilder outputComputerBuilder = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel);
        if (dropOutCoeffs != null) {
            outputComputerBuilder.withDropOutParameters(dropOutCoeffs);
        }
        return outputComputerBuilder.buildFinalOutputComputer();
    }

}
