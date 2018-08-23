package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.cost.CostType;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout.GradientDescentWithDerivationAndDropOutRegularizationProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout.GradientDescentWithDropOutRegularizationProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2.GradientDescentWithDerivationAndL2RegularizationProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2.GradientDescentWithL2RegularizationProcessProvider;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

class LearningAlgorithmBuilderTest {

    private NeuralNetworkModel buildNeuralNetworkModel() {
        return NeuralNetworkModelBuilder.init()
                .input(1)
                .addLayer(1)
                .buildWeightBiasModel();
    }

    @Nested
    class ParameterChecks {

        @Test
        void test_exception_if_neuralNetworkModel_is_missing() {
            Assertions.assertThrows(IllegalArgumentException.class, () -> LearningAlgorithmBuilder.init().build());
        }

        @Test
        void test_exception_if_many_regularization_methods() {
            Assertions.assertThrows(IllegalArgumentException.class,
                    () -> LearningAlgorithmBuilder.init()
                            .withModel(buildNeuralNetworkModel())
                            .withL2Regularization(0.5)
                            .withDropOutRegularitation()
                            .build()
            );
        }

        @Test
        void test_exception_if_drop_out_regularization_parameters_are_missing() {
            Assertions.assertThrows(IllegalArgumentException.class,
                    () -> LearningAlgorithmBuilder.init()
                            .withModel(buildNeuralNetworkModel())
                            .withDropOutRegularitation()
                            .build()
            );
        }

        @Test
        void test_exception_if_drop_out_regularization_parameters_are_incorrect() {
            Assertions.assertThrows(IllegalArgumentException.class,
                    () -> LearningAlgorithmBuilder.init()
                            .withModel(buildNeuralNetworkModel())
                            .withDropOutRegularitation(-0.5)
                            .build()
            );

            Assertions.assertThrows(IllegalArgumentException.class,
                    () -> LearningAlgorithmBuilder.init()
                            .withModel(buildNeuralNetworkModel())
                            .withDropOutRegularitation(1.5)
                            .build()
            );
        }
    }

    @Nested
    class GradientDescentInstanceCreation {

        @Test
        void test_GradientDescent_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentOnLinearRegressionProcessProvider);
        }

        @Test
        void test_GradientDescentWithL2Regularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT)
                    .withL2Regularization(0.5)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithL2RegularizationProcessProvider);
        }

        @Test
        void test_GradientDescentWithDropOutRegularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT)
                    .withDropOutRegularitation(0.1, 0.2)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDropOutRegularizationProcessProvider);
        }

    }

    @Nested
    class GradientDescentWithDerivationInstanceCreation {

        @Test
        void test_GradientDescentWithDerivation_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivation);
            Assertions.assertTrue(((GradientDescentWithDerivation) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDerivationProcessProvider);
        }

        @Test
        void test_GradientDescentWithDerivationAndL2Regularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION)
                    .withL2Regularization(0.5)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivation);
            Assertions.assertTrue(((GradientDescentWithDerivation) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDerivationAndL2RegularizationProcessProvider);
        }

        @Test
        void test_GradientDescentWithDerivationAndDropOutRegularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION)
                    .withDropOutRegularitation(0.1, 0.2)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivation);
            Assertions.assertTrue(((GradientDescentWithDerivation) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDerivationAndDropOutRegularizationProcessProvider);
        }
    }

    @Nested
    class GradientDescentWithOptimisationInstanceCreation {

        @Test
        void test_GradientDescentWithMomentum_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT)
                    .withMomentumCoeff(0.9)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithMomentumProcessProvider);
        }

        @Test
        void test_GradientDescentWithDerivationAndWithMomentum_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION)
                    .withMomentumCoeff(0.9)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivation);
            Assertions.assertTrue(((GradientDescentWithDerivation) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDerivationAndMomentumProcessProvider);
        }

        @Test
        void test_GradientDescentWithRmsStop_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT)
                    .withRmsStopCoeff(0.999)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithRmsStopProcessProvider);
        }

        @Test
        void test_GradientDescentWithDerivationAndRmsStop_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION)
                    .withRmsStopCoeff(0.999)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivation);
            Assertions.assertTrue(((GradientDescentWithDerivation) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDerivationAndRmsStopProcessProvider);
        }

        @Test
        void test_GradientDescentWithDerivationAndAdamOptimisation_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION)
                    .withMomentumCoeff(0.9)
                    .withRmsStopCoeff(0.999)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivation);
            Assertions.assertTrue(((GradientDescentWithDerivation) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDerivationAndAdamOptimisationProcessProvider);
        }

        @Test
        void test_GradientDescentWithAdamOptimisation_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT)
                    .withMomentumCoeff(0.9)
                    .withRmsStopCoeff(0.999)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithAdamOptimisationProcessProvider);
        }
    }

    @Nested
    class GradientDescentSoftMaxInstanceCreation {

        @Test
        void test_GradientDescentWithDerivation_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION)
                    .withCostType(CostType.SOFT_MAX_REGRESSION)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivation);
            Assertions.assertTrue(((GradientDescentWithDerivation) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDerivationProcessProvider);
        }

        @Test
        void test_GradientDescent_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT)
                    .withCostType(CostType.SOFT_MAX_REGRESSION)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentOnSoftMaxRegressionProcessProvider);
        }
    }

    @Nested
    class GradientDescentLogisticInstanceCreation {

        @Test
        void test_GradientDescentWithDerivation_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION)
                    .withCostType(CostType.LOGISTIC_REGRESSION)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivation);
            Assertions.assertTrue(((GradientDescentWithDerivation) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDerivationProcessProvider);
        }

        @Test
        void test_GradientDescent_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT)
                    .withCostType(CostType.LOGISTIC_REGRESSION)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentOnLogisticRegressionProcessProvider);
        }
    }
}