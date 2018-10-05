package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.computer.data.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.data.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.data.WeightBiasData;

public class InputCountVisitor implements DataVisitor {

    private int inputCount;

    @Override
    public void visit(WeightBiasData data) {
        inputCount = data.getData().getColumns();
    }

    @Override
    public void visit(BatchNormData data) {
        inputCount = data.getData().getColumns();
    }

    public int getInputCount() {
        return inputCount;
    }
}
