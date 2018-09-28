package com.fgiannesini.neuralnetwork.batch;

import com.fgiannesini.neuralnetwork.computer.WeightBiasData;
import org.jblas.DoubleMatrix;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.stream.IntStream;

class BatchIteratorTest {

    @Test
    void divided_in_3_last_batch_incomplete() {
        WeightBiasData input = new WeightBiasData(new DoubleMatrix(3, 10, IntStream.range(0, 30).asDoubleStream().toArray()));
        WeightBiasData output = new WeightBiasData(new DoubleMatrix(2, 10, IntStream.range(0, 20).asDoubleStream().toArray()));
        BatchIterator batchIterator = BatchIterator.init(input, output, 4);

        Assertions.assertTrue(batchIterator.hasNext());

        checkBatchMatrix(((WeightBiasData) batchIterator.getSubInput()).getInput(), new double[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}, 4, 3);
        checkBatchMatrix(((WeightBiasData) batchIterator.getSubOutput()).getInput(), new double[]{0, 1, 2, 3, 4, 5, 6, 7}, 4, 2);

        batchIterator.next();
        Assertions.assertTrue(batchIterator.hasNext());

        checkBatchMatrix(((WeightBiasData) batchIterator.getSubInput()).getInput(), new double[]{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}, 4, 3);
        checkBatchMatrix(((WeightBiasData) batchIterator.getSubOutput()).getInput(), new double[]{8, 9, 10, 11, 12, 13, 14, 15}, 4, 2);

        batchIterator.next();
        Assertions.assertTrue(batchIterator.hasNext());

        checkBatchMatrix(((WeightBiasData) batchIterator.getSubInput()).getInput(), new double[]{24, 25, 26, 27, 28, 29}, 2, 3);
        checkBatchMatrix(((WeightBiasData) batchIterator.getSubOutput()).getInput(), new double[]{16, 17, 18, 19}, 2, 2);

        batchIterator.next();
        Assertions.assertFalse(batchIterator.hasNext());
    }

    @Test
    void divided_in_2_all_batches_complete() {
        WeightBiasData input = new WeightBiasData(new DoubleMatrix(2, 10, IntStream.range(0, 20).asDoubleStream().toArray()));
        WeightBiasData output = new WeightBiasData(new DoubleMatrix(3, 10, IntStream.range(0, 30).asDoubleStream().toArray()));
        BatchIterator batchIterator = BatchIterator.init(input, output, 5);

        Assertions.assertTrue(batchIterator.hasNext());

        checkBatchMatrix(((WeightBiasData) batchIterator.getSubInput()).getInput(), new double[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 5, 2);
        checkBatchMatrix(((WeightBiasData) batchIterator.getSubOutput()).getInput(), new double[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}, 5, 3);

        batchIterator.next();
        Assertions.assertTrue(batchIterator.hasNext());

        checkBatchMatrix(((WeightBiasData) batchIterator.getSubInput()).getInput(), new double[]{10, 11, 12, 13, 14, 15, 16, 17, 18, 19}, 5, 2);
        checkBatchMatrix(((WeightBiasData) batchIterator.getSubOutput()).getInput(), new double[]{15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29}, 5, 3);

        batchIterator.next();
        Assertions.assertFalse(batchIterator.hasNext());
    }

    @Test
    void one_batches_incomplete() {
        WeightBiasData input = new WeightBiasData(new DoubleMatrix(2, 5, IntStream.range(0, 10).asDoubleStream().toArray()));
        WeightBiasData output = new WeightBiasData(new DoubleMatrix(2, 5, IntStream.range(0, 10).asDoubleStream().toArray()));
        BatchIterator batchIterator = BatchIterator.init(input, output, 10);

        Assertions.assertTrue(batchIterator.hasNext());

        checkBatchMatrix(((WeightBiasData) batchIterator.getSubInput()).getInput(), new double[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 5, 2);
        checkBatchMatrix(((WeightBiasData) batchIterator.getSubOutput()).getInput(), new double[]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 5, 2);

        batchIterator.next();
        Assertions.assertFalse(batchIterator.hasNext());
    }

    private void checkBatchMatrix(DoubleMatrix batchMatrix, double[] expectedData, int expectedColumnsCount, int expectedRowsCount) {
        Assertions.assertEquals(expectedRowsCount, batchMatrix.getRows());
        Assertions.assertEquals(expectedColumnsCount, batchMatrix.getColumns());
        Assertions.assertArrayEquals(expectedData, batchMatrix.data);
    }

}