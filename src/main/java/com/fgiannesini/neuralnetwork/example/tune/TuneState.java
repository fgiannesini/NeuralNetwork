package com.fgiannesini.neuralnetwork.example.tune;

import com.fgiannesini.neuralnetwork.HyperParameters;

public class TuneState {

    private double mark;
    private double successRate;
    private double executionTime;
    private final HyperParameters hyperParameters;

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

    public double getSuccessRate() {
        return successRate;
    }

    public void setSuccessRate(double successRate) {
        this.successRate = successRate;
    }

    public double getExecutionTime() {
        return executionTime;
    }

    public void setExecutionTime(double executionTime) {
        this.executionTime = executionTime;
    }
}
