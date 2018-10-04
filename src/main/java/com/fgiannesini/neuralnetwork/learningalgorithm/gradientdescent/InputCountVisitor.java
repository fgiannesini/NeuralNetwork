package com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent;

import com.fgiannesini.neuralnetwork.computer.BatchNormData;
import com.fgiannesini.neuralnetwork.computer.DataVisitor;
import com.fgiannesini.neuralnetwork.computer.WeightBiasData;

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
