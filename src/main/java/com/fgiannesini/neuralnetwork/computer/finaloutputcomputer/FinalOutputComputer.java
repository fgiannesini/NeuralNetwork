package com.fgiannesini.neuralnetwork.computer.finaloutputcomputer;

import com.fgiannesini.neuralnetwork.computer.DataFunctionApplier;
import com.fgiannesini.neuralnetwork.computer.LayerComputerVisitor;
import com.fgiannesini.neuralnetwork.computer.LayerTypeData;
import com.fgiannesini.neuralnetwork.computer.intermediateoutputcomputer.IntermediateOutputResult;
import com.fgiannesini.neuralnetwork.model.Layer;
import org.jblas.DoubleMatrix;

import java.util.List;

public class FinalOutputComputer implements IFinalOutputComputer {

    private List<Layer> layers;

    public FinalOutputComputer(List<Layer> layers) {
        this.layers = layers;
    }

    @Override
    public LayerTypeData compute(LayerTypeData input) {
        LayerTypeData firstData = input.accept(new DataFunctionApplier(DoubleMatrix::dup));
        IntermediateOutputResult intermediateOutputResult = new IntermediateOutputResult(firstData);
        for (Layer layer : layers) {
            LayerComputerVisitor layerComputerVisitor = new LayerComputerVisitor(intermediateOutputResult.getResult());
            layer.accept(layerComputerVisitor);
            intermediateOutputResult = layerComputerVisitor.getIntermediateOutputResult();
        }
        return intermediateOutputResult.getResult();
    }

}
