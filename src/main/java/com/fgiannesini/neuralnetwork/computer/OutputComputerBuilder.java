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

    public OutputComputerBuilder<L> withModel(NeuralNetworkModel<L> neuralNetworkModel) {
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
            ILayerComputer<L> layerComputer = LayerComputerBuilder.init()
                    .withLayerType(neuralNetworkModel.getLayerType())
                    .build();
            finalOutputComputer = new FinalOutputComputerWithDropOutRegularization<L>(dropOutMatrixList, layerComputer, neuralNetworkModel.getLayers());
        } else {
            ILayerComputer<L> layerComputer = LayerComputerBuilder.init()
                    .withLayerType(neuralNetworkModel.getLayerType())
                    .build();
            finalOutputComputer = new FinalOutputComputer<L>(neuralNetworkModel.getLayers(), layerComputer);
        }
        return finalOutputComputer;
    }

    public IIntermediateOutputComputer buildIntermediateOutputComputer() {
        checkParameters();
        IIntermediateOutputComputer intermediateOutputComputer;
        if (dropOutMatrixList != null) {
            ILayerComputer<L> layerComputer = LayerComputerBuilder.init()
                    .withLayerType(neuralNetworkModel.getLayerType())
                    .build();
            intermediateOutputComputer = new IntermediateOutputComputerWithDropOutRegularization<>(dropOutMatrixList, layerComputer, neuralNetworkModel.getLayers());
        } else {
            ILayerComputer<L> layerComputer = LayerComputerBuilder.init()
                    .withLayerType(neuralNetworkModel.getLayerType())
                    .build();
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
