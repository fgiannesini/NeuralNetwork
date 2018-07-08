package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.FinalOutputComputer;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.FinalOutputComputerWithDropOutRegularization;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IIntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;

public class OutputComputerBuilder {

    private NeuralNetworkModel neuralNetworkModel;

    private List<DoubleMatrix> dropOutMatrixList;

    private OutputComputerBuilder() {
    }

    public static OutputComputerBuilder init() {
        return new OutputComputerBuilder();
    }

    public OutputComputerBuilder withModel(NeuralNetworkModel neuralNetworkModel) {
        this.neuralNetworkModel = neuralNetworkModel;
        return this;
    }

    public OutputComputerBuilder withDropOutParameters(List<DoubleMatrix> dropOutMatrixList) {
        this.dropOutMatrixList = dropOutMatrixList;
        return this;
    }

    public IFinalOutputComputer buildFinalOutputComputer() {
        checkParameters();
        IFinalOutputComputer finalOutputComputer;
        if (dropOutMatrixList != null) {
            finalOutputComputer = new FinalOutputComputerWithDropOutRegularization(neuralNetworkModel, dropOutMatrixList);
        } else {
            finalOutputComputer = new FinalOutputComputer(neuralNetworkModel);
        }
        return finalOutputComputer;
    }

    public IIntermediateOutputComputer buildIntermediateOutputComputer() {
        checkParameters();
        IIntermediateOutputComputer intermediateOutputComputer;
        if (dropOutMatrixList != null) {
            throw new UnsupportedOperationException();
        } else {
            intermediateOutputComputer = new IntermediateOutputComputer(neuralNetworkModel);
        }

        return intermediateOutputComputer;
    }

    private void checkParameters() {
        if (neuralNetworkModel == null) {
            throw new IllegalArgumentException("Missing neural Network");
        }
    }
}
