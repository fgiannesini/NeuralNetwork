package com.fgiannesini.neuralnetwork.learningalgorithm;

import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout.GradientDescentWithDerivationAndDropOutRegularization;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.dropout.GradientDescentWithDropOutRegularization;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2.GradientDescentWithDerivationAndL2Regularization;
import com.fgiannesini.neuralnetwork.learningalgorithm.regularization.l2.GradientDescentWithL2Regularization;
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
    public class ParameterChecks {

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
    public class InstanceCreation {

        @Test
        void test_GradientDescent_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescent);
        }

        @Test
        void test_GradientDescentWithL2Regularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT)
                    .withL2Regularization(0.5)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithL2Regularization);
        }

        @Test
        void test_GradientDescentWithDropOutRegularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT)
                    .withDropOutRegularitation(0.1, 0.2)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDropOutRegularization);
        }

        @Test
        void test_GradientDescentWithDerivation_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivation);
        }

        @Test
        void test_GradientDescentWithDerivationAndL2Regularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION)
                    .withL2Regularization(0.5)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivationAndL2Regularization);
        }

        @Test
        void test_GradientDescentWithDerivationAndDropOutRegularization_instance_creation() {
            LearningAlgorithm learningAlgorithm = LearningAlgorithmBuilder.init()
                    .withModel(buildNeuralNetworkModel())
                    .withAlgorithmType(LearningAlgorithmType.GRADIENT_DESCENT_DERIVATION)
                    .withDropOutRegularitation(0.1, 0.2)
                    .build();

            Assertions.assertTrue(learningAlgorithm instanceof GradientDescentWithDerivationAndDropOutRegularization);
        }
    }
}