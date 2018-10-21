package com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer;

import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.data.adapter.ForwardDataAdapterVisitor;
import com.fgiannesini.neuralnetwork.computer.finaloutputcomputer.LayerComputerWithDropOutRegularizationVisitor;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.ArrayList;
import java.util.List;

public class IntermediateOutputComputerWithDropOutRegularization implements IIntermediateOutputComputer {

    private final List<DoubleMatrix> dropOutMatrixList;
    private final List<Layer> layers;

    public IntermediateOutputComputerWithDropOutRegularization(List<DoubleMatrix> dropOutMatrixList, List<Layer> layers) {
        this.dropOutMatrixList = dropOutMatrixList;
        this.layers = layers;
    }

    public List<IntermediateOutputResult> compute(LayerTypeData data) {
        List<IntermediateOutputResult> intermediateOutputResults = new ArrayList<>();

        InputDropOutRegularizationVisitor inputVisitor = new InputDropOutRegularizationVisitor(dropOutMatrixList.get(0));
        data.accept(inputVisitor);
        LayerTypeData regularizedInput = inputVisitor.getRegularizedData();
        ForwardDataAdapterVisitor firstDataAdaptorVisitor = new ForwardDataAdapterVisitor(regularizedInput);
        layers.get(0).accept(firstDataAdaptorVisitor);
        IntermediateOutputResult intermediateOutputResult = new IntermediateOutputResult(firstDataAdaptorVisitor.getData());
        intermediateOutputResults.add(intermediateOutputResult);

        for (int layerIndex = 0, dropOutIndex = 1; layerIndex < layers.size(); layerIndex++, dropOutIndex++) {
            Layer layer = layers.get(layerIndex);
            LayerTypeData previousResult = intermediateOutputResult.getResult();
            ForwardDataAdapterVisitor dataAdaptorVisitor = new ForwardDataAdapterVisitor(previousResult);
            layer.accept(dataAdaptorVisitor);
            LayerComputerWithDropOutRegularizationVisitor computerVisitor = new LayerComputerWithDropOutRegularizationVisitor(dropOutMatrixList.get(dropOutIndex), dataAdaptorVisitor.getData());
            layer.accept(computerVisitor);
            intermediateOutputResult = computerVisitor.getIntermediateOutputResult();
            intermediateOutputResults.add(intermediateOutputResult);
        }
        return intermediateOutputResults;
    }

}
