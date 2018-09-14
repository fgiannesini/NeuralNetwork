package com.fgiannesini.neuralnetwork.computer;

import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.FinalOutputComputer;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.FinalOutputComputerWithDropOutRegularization;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.IFinalOutputComputer;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IIntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputComputer;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputComputerWithDropOutRegularization;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.List;

public class OutputComputerBuilder<L extends Layer> {

    private NeuralNetworkModel<L> neuralNetworkModel;

    private List<DoubleMatrix> dropOutMatrixList;

    private OutputComputerBuilder() {
    }

    public static OutputComputerBuilder init() {
        return new OutputComputerBuilder<>();
    }

    public OutputComputerBuilder withModel(NeuralNetworkModel<L> neuralNetworkModel) {
        this.neuralNetworkModel = neuralNetworkModel;
        return this;
    }

    public OutputComputerBuilder withDropOutParameters(List<DoubleMatrix> dropOutMatrixList) {
        this.dropOutMatrixList = dropOutMatrixList;
        return this;
    }

    public IFinalOutputComputer<L> buildFinalOutputComputer() {
        checkParameters();
        ILayerComputer<L> layerComputer = LayerComputerBuilder.init()
                .withLayerType(neuralNetworkModel.getLayerType())
                .build();
        IFinalOutputComputer<L> finalOutputComputer;
        if (dropOutMatrixList != null) {
            finalOutputComputer = new FinalOutputComputerWithDropOutRegularization<>(dropOutMatrixList, layerComputer, neuralNetworkModel.getLayers());
        } else {
            finalOutputComputer = new FinalOutputComputer<>(neuralNetworkModel.getLayers(), layerComputer);
        }
        return finalOutputComputer;
    }

    public IIntermediateOutputComputer<L> buildIntermediateOutputComputer() {
        checkParameters();
        ILayerComputer<L> layerComputer = LayerComputerBuilder.init()
                .withLayerType(neuralNetworkModel.getLayerType())
                .build();
        IIntermediateOutputComputer<L> intermediateOutputComputer;
        if (dropOutMatrixList != null) {
            intermediateOutputComputer = new IntermediateOutputComputerWithDropOutRegularization<>(dropOutMatrixList, layerComputer, neuralNetworkModel.getLayers());
        } else {
            intermediateOutputComputer = new IntermediateOutputComputer<>(neuralNetworkModel, layerComputer);
        }

        return intermediateOutputComputer;
    }

    private void checkParameters() {
        if (neuralNetworkModel == null) {
            throw new IllegalArgumentException("Missing neural Network");
        }
    }
}
