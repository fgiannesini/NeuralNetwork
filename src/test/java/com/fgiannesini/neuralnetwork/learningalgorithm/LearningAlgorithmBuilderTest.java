package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.*;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivation;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivationAndMomentumProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivationAndRmsStopProcessProvider;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescentwithderivation.GradientDescentWithDerivationProcessProvider;
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
                .build();
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
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentProcessProvider);
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
    class GradientDescentWithMomentumInstanceCreation {

        @Test
        void test_GradientDescentWithMomentum_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_MOMENTUM)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithMomentumProcessProvider);
        }

        @Test
        void test_GradientDescentWithMomentumAndL2Regularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_MOMENTUM)
                    .withL2Regularization(0.5)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithL2RegularizationProcessProvider);
        }

        @Test
        void test_GradientDescentWithMomentumAndDropOutRegularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_MOMENTUM)
                    .withDropOutRegularitation(0.1, 0.2)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDropOutRegularizationProcessProvider);
        }
    }

    @Nested
    class GradientDescentWithDerivationAndMomentumInstanceCreation {

        @Test
        void test_GradientDescentWithMomentum_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION_MOMENTUM)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivation);
            Assertions.assertTrue(((GradientDescentWithDerivation) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDerivationAndMomentumProcessProvider);
        }

        @Test
        void test_GradientDescentWithMomentumAndL2Regularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION_MOMENTUM)
                    .withL2Regularization(0.5)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivation);
            Assertions.assertTrue(((GradientDescentWithDerivation) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDerivationAndL2RegularizationProcessProvider);
        }

        @Test
        void test_GradientDescentWithMomentumAndDropOutRegularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION_MOMENTUM)
                    .withDropOutRegularitation(0.1, 0.2)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivation);
            Assertions.assertTrue(((GradientDescentWithDerivation) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDerivationAndDropOutRegularizationProcessProvider);
        }
    }

    @Nested
    class GradientDescentWithRmsStopInstanceCreation {

        @Test
        void test_GradientDescentWithRmsStop_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_RMS_STOP)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithRmsStopProcessProvider);
        }

        @Test
        void test_GradientDescentWithRmsStopAndL2Regularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_RMS_STOP)
                    .withL2Regularization(0.5)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithL2RegularizationProcessProvider);
        }

        @Test
        void test_GradientDescentWithRmsStopAndDropOutRegularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_RMS_STOP)
                    .withDropOutRegularitation(0.1, 0.2)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDropOutRegularizationProcessProvider);
        }
    }


    @Nested
    class GradientDescentWithDerivationAndRmsStopInstanceCreation {

        @Test
        void test_GradientDescentWithDerivationAndRmsStop_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION_RMS_STOP)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivation);
            Assertions.assertTrue(((GradientDescentWithDerivation) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDerivationAndRmsStopProcessProvider);
        }

        @Test
        void test_GradientDescentWithDerivationAndRmsStopAndL2Regularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION_RMS_STOP)
                    .withL2Regularization(0.5)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivation);
            Assertions.assertTrue(((GradientDescentWithDerivation) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDerivationAndL2RegularizationProcessProvider);
        }

        @Test
        void test_GradientDescentWithDerivationAndRmsStopAndDropOutRegularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION_RMS_STOP)
                    .withDropOutRegularitation(0.1, 0.2)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivation);
            Assertions.assertTrue(((GradientDescentWithDerivation) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDerivationAndDropOutRegularizationProcessProvider);
        }
    }


    @Nested
    class GradientDescentWithAdamOptimisationInstanceCreation {

        @Test
        void test_GradientDescentWithAdamOptimisation_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_ADAM_OPTIMISATION)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithAdamOptimisationProcessProvider);
        }

        @Test
        void test_GradientDescentWithAdamOptimisationAndL2Regularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_ADAM_OPTIMISATION)
                    .withL2Regularization(0.5)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithL2RegularizationProcessProvider);
        }

        @Test
        void test_GradientDescentWithAdamOptimisationAndDropOutRegularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_ADAM_OPTIMISATION)
                    .withDropOutRegularitation(0.1, 0.2)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
            Assertions.assertTrue(((GradientDescent) learningAlgorithm).getGradientDescentProcessProvider() instanceof GradientDescentWithDropOutRegularizationProcessProvider);
        }
    }

}