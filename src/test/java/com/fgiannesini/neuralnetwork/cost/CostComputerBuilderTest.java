package com.fgiannesini.neuralnetwork.cost;

import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class CostComputerBuilderTest {

    @Test
    void check_neural_network_is_mandatory() {
        Assertions.assertThrows(IllegalArgumentException.class, () -> CostComputerBuilder.init().build());
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

        Assertions.assertTrue(costComputer instanceof CostComputerWithL2LinearRegression);
    }

    @Test
    void check_Logistic_Regression_Cost_Computer() {
        CostComputer costComputer = CostComputerBuilder.init()
                .withNeuralNetworkModel(buildNeuralNetworkModel())
                .withType(CostType.LOGISTIC_REGRESSION)
                .build();

        Assertions.assertTrue(costComputer instanceof LogisticRegressionCostComputer);
    }

    private NeuralNetworkModel buildNeuralNetworkModel() {
        return NeuralNetworkModelBuilder.init()
                .input(1)
                .addLayer(1)
                .build();
    }

}