package com.fgiannesini.neuralnetwork.batch;

import com.fgiannesini.neuralnetwork.computer.data.LayerTypeData;
import com.fgiannesini.neuralnetwork.learningalgorithm.gradientdescent.InputCountVisitor;

public class BatchIterator {

    private final int batchSize;
    private final LayerTypeData input;
    private final LayerTypeData output;
    private final int inputCount;
    private int lowerIndex;
    private int upperIndex;
    private int batchNumber;

    private BatchIterator(LayerTypeData input, LayerTypeData output, int batchSize) {
        this.batchSize = batchSize;
        this.input = input;
        this.output = output;
        lowerIndex = 0;
        InputCountVisitor countVisitor = new InputCountVisitor();
        input.accept(countVisitor);
        inputCount = countVisitor.getInputCount();
        upperIndex = Math.min(batchSize, inputCount);
        batchNumber = 1;
    }

    public static BatchIterator init(LayerTypeData input, LayerTypeData output, int batchSize) {
        return new BatchIterator(input, output, batchSize);
    }

    public boolean hasNext() {
        return lowerIndex == 0 || lowerIndex < inputCount;
    }

    public void next() {
        lowerIndex += batchSize;
        upperIndex = Math.min(upperIndex + batchSize, inputCount);
        batchNumber++;
    }

    public LayerTypeData getSubInput() {
        BatchIteratorVisitor batchIteratorVisitor = new BatchIteratorVisitor(lowerIndex, upperIndex);
        input.accept(batchIteratorVisitor);
        return batchIteratorVisitor.getSubData();
    }

    public LayerTypeData getSubOutput() {
        BatchIteratorVisitor batchIteratorVisitor = new BatchIteratorVisitor(lowerIndex, upperIndex);
        output.accept(batchIteratorVisitor);
        return batchIteratorVisitor.getSubData();
    }

    public int getBatchNumber() {
        return batchNumber;
    }
}
