package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.ILayerComputer;
import com.fgiannesini.neuralnetwork.computer.LayerComputerBuilder;
import com.fgiannesini.neuralnetwork.model.Layer;
import com.fgiannesini.neuralnetwork.model.NeuralNetworkModel;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class IntermediateOutputComputerWithDropOutRegularization implements IIntermediateOutputComputer {

    private final NeuralNetworkModel<? extends Layer> model;
    private final List<DoubleMatrix> dropOutMatrixList;
    private final ILayerComputer layerComputer;

    public IntermediateOutputComputerWithDropOutRegularization(NeuralNetworkModel<? extends Layer> model, List<DoubleMatrix> dropOutMatrixList) {
        this.model = model;
        layerComputer = LayerComputerBuilder.init()
                .withLayerType(model.getLayerType())
                .build();
        this.dropOutMatrixList = dropOutMatrixList;
    }

    public List<DoubleMatrix> compute(DoubleMatrix inputMatrix) {
        List<DoubleMatrix> intermediateMatrix = new ArrayList<>();
        DoubleMatrix currentMatrix = inputMatrix.dup().muliColumnVector(dropOutMatrixList.get(0));
        intermediateMatrix.add(currentMatrix);
        for (int layerIndex = 0, dropOutIndex = 1; layerIndex < model.getLayers().size(); layerIndex++, dropOutIndex++) {
            Layer layer = model.getLayers().get(layerIndex);
            currentMatrix = layerComputer.computeZFromInput(currentMatrix, layer);
            currentMatrix.muliColumnVector(dropOutMatrixList.get(dropOutIndex));
            currentMatrix = layerComputer.computeAFromZ(currentMatrix, layer);
            intermediateMatrix.add(currentMatrix);
        }
        return intermediateMatrix;
    }

}
