package com.fgiannesini.neuralnetwork.example.tune;

import com.fgiannesini.neuralnetwork.HyperParameters;

public class TuneState {

    private double mark;
    private HyperParameters hyperParameters;

    public TuneState(HyperParameters hyperParameters) {
        this.hyperParameters = hyperParameters;
    }

    public double getMark() {
        return mark;
    }

    public void setMark(double mark) {
        this.mark = mark;
    }

    public HyperParameters getHyperParameters() {
        return hyperParameters;
    }
}
