package com.fgiannesini.neuralnetwork.activationfunctions;

public enum ActivationFunctionType {

    NONE {
        @Override
        public ActivationFunctionApplier getActivationFunction() {
            return matrix -> matrix;
        }
    },
    SIGMOID {
        @Override
        public ActivationFunctionApplier getActivationFunction() {
            return new SigmoidFunctionApplier();
        }
    },
    TANH {
        @Override
        public ActivationFunctionApplier getActivationFunction() {
            return new TanhFunctionApplier();
        }
    },
    RELU {
        @Override
        public ActivationFunctionApplier getActivationFunction() {
            return new ReluFunctionApplier();
        }
    },
    LEAKY_RELU {
        @Override
        public ActivationFunctionApplier getActivationFunction() {
            return new LeakyReluFunctionApplier();
        }
    };

    public abstract ActivationFunctionApplier getActivationFunction();
}
