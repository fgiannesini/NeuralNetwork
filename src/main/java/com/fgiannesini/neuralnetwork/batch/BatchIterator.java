package com.fgiannesini.neuralnetwork.batch;

import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

public class BatchIterator {

    private int batchSize;
    private DoubleMatrix input;
    private DoubleMatrix output;
    private int lowerIndex;
    private int upperIndex;
    private int batchNumber;

    private BatchIterator(DoubleMatrix input, DoubleMatrix output, int batchSize) {
        this.batchSize = batchSize;
        this.input = input;
        this.output = output;
        lowerIndex = 0;
        upperIndex = Math.min(batchSize, input.getColumns());
        batchNumber = 1;
    }

    public static BatchIterator init(DoubleMatrix input, DoubleMatrix output, int batchSize) {
        return new BatchIterator(input, output, batchSize);
    }

    public boolean hasNext() {
        return lowerIndex == 0 || lowerIndex < input.getColumns();
    }

    public void next() {
        lowerIndex += batchSize;
        upperIndex = Math.min(upperIndex + batchSize, input.getColumns());
        batchNumber++;
    }

    public DoubleMatrix getSubInput() {
        return input.getColumns(new IntervalRange(lowerIndex, upperIndex));
    }

    public DoubleMatrix getSubOutput() {
        return output.getColumns(new IntervalRange(lowerIndex, upperIndex));
    }

    public int getBatchNumber() {
        return batchNumber;
    }
}
