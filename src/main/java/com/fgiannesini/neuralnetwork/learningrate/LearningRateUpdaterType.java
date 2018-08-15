package com.fgiannesini.neuralnetwork.learningrate;

public enum LearningRateUpdaterType {

    CONSTANT {
        @Override
        public ILearningRateUpdater get(double initialLearningRate) {
            return new ConstantLearningRateUpdater(initialLearningRate);
        }
    },
    EXPONENTIALLY {
        @Override
        public ILearningRateUpdater get(double initialLearningRate) {
            return new ExponentiallyLearningRateUpdater(initialLearningRate);
        }
    },
    SQUARED {
        @Override
        public ILearningRateUpdater get(double initialLearningRate) {
            return new SquaredLearningRateUpdater(initialLearningRate);
        }
    },
    DECAY {
        @Override
        public ILearningRateUpdater get(double initialLearningRate) {
            return new DecayLearningRateUpdater(initialLearningRate);
        }
    };

    public abstract ILearningRateUpdater get(double initialLearningRate);
}
