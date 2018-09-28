package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;

class CostComputerBuilderTest {

    @Test
    void check_neural_network_is_mandatory() {
        Assertions.assertThrows(IllegalArgumentException.class, () -> CostComputerBuilder.init().build());
    }

    @Test
    void check_exception_if_many_regularization_method() {
        Assertions.assertThrows(IllegalArgumentException.class,
                () -> CostComputerBuilder.init()
                        .withNeuralNetworkModel(buildNeuralNetworkModel())
                        .withL2Regularization(0.5)
                        .withDropOutRegularization(new ArrayList<>())
                        .build());
    }

    @Test
    void check_Linear_Regression_Cost_Computer() {
        CostComputer costComputer = CostComputerBuilder.init()
                .withNeuralNetworkModel(buildNeuralNetworkModel())
                .withType(CostType.LINEAR_REGRESSION)
                .build();

        Assertions.assertTrue(costComputer instanceof LinearRegressionCostComputer);
    }

    @Test
    void check_L2Regularization_Cost_Computer() {
        CostComputer costComputer = CostComputerBuilder.init()
                .withNeuralNetworkModel(buildNeuralNetworkModel())
                .withType(CostType.LINEAR_REGRESSION)
                .withL2Regularization(0.5)
                .build();

        Assertions.assertTrue(costComputer instanceof CostComputerWithL2Regularization);
    }

    @Test
    void check_Logistic_Regression_Cost_Computer() {
        CostComputer costComputer = CostComputerBuilder.init()
                .withNeuralNetworkModel(buildNeuralNetworkModel())
                .withType(CostType.LOGISTIC_REGRESSION)
                .build();

        Assertions.assertTrue(costComputer instanceof LogisticRegressionCostComputer);
    }

    @Test
    void check_SoftMax_Regression_Cost_Computer() {
        CostComputer costComputer = CostComputerBuilder.init()
                .withNeuralNetworkModel(buildNeuralNetworkModel())
                .withType(CostType.SOFT_MAX_REGRESSION)
                .build();

        Assertions.assertTrue(costComputer instanceof SoftMaxRegressionCostComputer);
    }

    private NeuralNetworkModel buildNeuralNetworkModel() {
        return NeuralNetworkModelBuilder.init()
                .input(1)
                .addWeightBiasLayer(1, ActivationFunctionType.RELU)
                .buildNeuralNetworkModel();
    }

}