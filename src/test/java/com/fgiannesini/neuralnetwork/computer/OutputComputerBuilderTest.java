package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.activationfunctions.ActivationFunctionType;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.FinalOutputComputer;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.FinalOutputComputerWithDropOutRegularization;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IIntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputComputerWithDropOutRegularization;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModelBuilder;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.Collections;

class OutputComputerBuilderTest {

    @Test
    void test_exception_if_neuralNetworkModel_missing() {
        Assertions.assertThrows(IllegalArgumentException.class, () -> OutputComputerBuilder.init().buildFinalOutputComputer());
        Assertions.assertThrows(IllegalArgumentException.class, () -> OutputComputerBuilder.init().buildIntermediateOutputComputer());
    }

    @Test
    void test_FinalOutputComputer_instance_creation() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .input(1)
                .addWeightBiasLayer(1, ActivationFunctionType.RELU)
                .buildNeuralNetworkModel();

        IFinalOutputComputer finalOutputComputer = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .buildFinalOutputComputer();

        Assertions.assertTrue(finalOutputComputer instanceof FinalOutputComputer);
    }

    @Test
    void test_FinalOutputComputerWithDropOutRegularization_instance_creation() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .input(1)
                .addWeightBiasLayer(1, ActivationFunctionType.RELU)
                .buildNeuralNetworkModel();

        IFinalOutputComputer finalOutputComputer = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .withDropOutParameters(Collections.singletonList(DoubleMatrix.EMPTY))
                .buildFinalOutputComputer();

        Assertions.assertTrue(finalOutputComputer instanceof FinalOutputComputerWithDropOutRegularization);
    }

    @Test
    void test_IntermediateOutputComputer_instance_creation() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .input(1)
                .addWeightBiasLayer(1, ActivationFunctionType.RELU)
                .buildNeuralNetworkModel();

        IIntermediateOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .buildIntermediateOutputComputer();

        Assertions.assertTrue(outputComputer instanceof IntermediateOutputComputer);
    }

    @Test
    void test_IntermediateOutputComputerWithDropOutRegularization_instance_creation() {
        NeuralNetworkModel neuralNetworkModel = NeuralNetworkModelBuilder.init()
                .input(1)
                .addWeightBiasLayer(1, ActivationFunctionType.RELU)
                .buildNeuralNetworkModel();

        IIntermediateOutputComputer outputComputer = OutputComputerBuilder.init()
                .withModel(neuralNetworkModel)
                .withDropOutParameters(Collections.singletonList(DoubleMatrix.EMPTY))
                .buildIntermediateOutputComputer();

        Assertions.assertTrue(outputComputer instanceof IntermediateOutputComputerWithDropOutRegularization);
    }
}