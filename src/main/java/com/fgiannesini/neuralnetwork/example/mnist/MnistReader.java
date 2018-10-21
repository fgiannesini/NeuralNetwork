/*
The MIT License (MIT)

Copyright (c) 2017 Ralf Th. Pietsch <ratopi@abwesend.de>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
 */
package com.fgiannesini.neuralnetwork.example.mnist;

import com.fgiannesini.neuralnetwork.example.mnist.io.MnistImageProvider;
import com.fgiannesini.neuralnetwork.example.mnist.io.MnistLabelProvider;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

public class MnistReader {
    private MnistLabelProvider mnistLabelProvider;
    private MnistImageProvider mnistImageProvider;

    public MnistReader(final File mnistLabelFile, final File mnistImageFile) throws IOException {
        this(new MnistLabelProvider(mnistLabelFile), new MnistImageProvider(mnistImageFile));
    }

    public MnistReader(MnistLabelProvider mnistLabelProvider, MnistImageProvider mnistImageProvider) {
        this.mnistLabelProvider = mnistLabelProvider;
        this.mnistImageProvider = mnistImageProvider;

        if (mnistLabelProvider.getNumberOfItems() != mnistImageProvider.getNumberOfItems()) {
            throw new RuntimeException("The count of items differs");
        }
    }

    public void close() throws IOException {
        this.mnistLabelProvider.close();
        this.mnistImageProvider.close();
    }

    public int getNumberOfItems() {
        return this.mnistImageProvider.getNumberOfItems();
    }

    public BufferedImage getDataAsBufferedImage(final byte[] values) {
        return mnistImageProvider.getDataAsBufferedImage(values);
    }

    public int handleAllRemaining(final DataArrayImageHandler imageHandler) throws IOException {
        return handleSome(Integer.MAX_VALUE, imageHandler);
    }

    public int handleSome(final int countToHandle, final DataArrayImageHandler imageHandler) throws IOException {
        int handledCount = 0;
        while (handledCount < countToHandle && mnistImageProvider.hasNext() && mnistImageProvider.hasNext()) {
            mnistImageProvider.selectNext();
            mnistLabelProvider.selectNext();

            final byte item = mnistLabelProvider.getCurrentValue();
            imageHandler.handle(mnistImageProvider.currentIndex(), mnistImageProvider.getCurrentData(), item);

            handledCount++;
        }

        mnistLabelProvider.close();
        mnistImageProvider.close();

        return handledCount;
    }

    // === public interfaces ===

    public interface DataArrayImageHandler {
        void handle(long index, byte[] data, final byte item);
    }

}
