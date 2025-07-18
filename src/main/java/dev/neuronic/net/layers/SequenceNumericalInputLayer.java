package dev.neuronic.net.layers;

import dev.neuronic.net.Shape;
import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.layers.Layer.LayerContext;
import dev.neuronic.net.math.FastRandom;
import dev.neuronic.net.optimizers.Optimizer;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Input layer for numerical time series data that reshapes and optionally scales values.
 * 
 * <p>Transforms a flat array of sequential values into a proper 2D sequence tensor
 * for RNN/GRU/LSTM processing, with optional scaling or normalization.
 * 
 * <p><b>Input:</b> [value1, value2, ..., valueN] - flat array of N values
 * <br><b>Output:</b> [N, 1] tensor - N timesteps, each with 1 feature
 * 
 * <p>This layer is essential for time series forecasting where you have a single
 * variable changing over time (revenue, temperature, stock price, etc.)
 */
public class SequenceNumericalInputLayer implements Layer {
    
    private final int sequenceLength;
    private final Feature feature;
    private final Optimizer optimizer;
    
    // For AUTO_NORMALIZE: track running statistics
    private final AtomicReference<NormalizationStats> stats;
    
    private static class NormalizationStats {
        volatile double mean = 0.0;
        volatile double variance = 1.0;
        volatile long count = 0;
    }
    
    public SequenceNumericalInputLayer(int sequenceLength, Feature feature, Optimizer optimizer) {
        this.sequenceLength = sequenceLength;
        this.feature = feature;
        this.optimizer = optimizer;
        this.stats = (feature.getType() == Feature.Type.AUTO_NORMALIZE) 
            ? new AtomicReference<>(new NormalizationStats()) 
            : null;
    }
    
    @Override
    public LayerContext forward(float[] input, boolean isTraining) {
        if (input.length != sequenceLength) {
            throw new IllegalArgumentException(
                "Expected input length " + sequenceLength + " but got " + input.length);
        }
        
        // Create output in sequence format [seqLen, 1]
        float[] output = new float[sequenceLength];
        
        // Apply feature transformation
        switch (feature.getType()) {
            case PASSTHROUGH:
                // Direct copy
                System.arraycopy(input, 0, output, 0, sequenceLength);
                break;
                
            case SCALE_BOUNDED:
                // Min-max scaling with user-specified bounds
                float min = feature.getMinBound();
                float max = feature.getMaxBound();
                float range = max - min;
                
                if (range == 0) {
                    // All values map to 0.5 if min == max
                    for (int i = 0; i < sequenceLength; i++) {
                        output[i] = 0.5f;
                    }
                } else {
                    for (int i = 0; i < sequenceLength; i++) {
                        float scaled = (input[i] - min) / range;
                        output[i] = Math.max(0.0f, Math.min(1.0f, scaled)); // Clamp to [0,1]
                    }
                }
                break;
                
            case AUTO_NORMALIZE:
                // Z-score normalization with running statistics
                updateStatistics(input);
                NormalizationStats currentStats = stats.get();
                double stdDev = Math.sqrt(currentStats.variance);
                
                if (stdDev < 1e-6) {
                    // Near-zero variance, output zeros
                    for (int i = 0; i < sequenceLength; i++) {
                        output[i] = 0.0f;
                    }
                } else {
                    for (int i = 0; i < sequenceLength; i++) {
                        output[i] = (float) ((input[i] - currentStats.mean) / stdDev);
                    }
                }
                break;
                
            default:
                throw new IllegalStateException("Unsupported feature type: " + feature.getType());
        }
        
        // Return with proper sequence shape
        return new LayerContext(input, null, output);
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient) {
        // For SCALE_BOUNDED and PASSTHROUGH, gradient flows through with scaling
        // For AUTO_NORMALIZE, gradient needs adjustment but we keep it simple for now
        float[] gradientOut = new float[sequenceLength];
        float[] gradientIn = upstreamGradient;
        
        switch (feature.getType()) {
            case PASSTHROUGH:
                // Direct gradient flow
                System.arraycopy(gradientIn, 0, gradientOut, 0, sequenceLength);
                break;
                
            case SCALE_BOUNDED:
                // Gradient scaled by 1/range
                float range = feature.getMaxBound() - feature.getMinBound();
                if (range != 0) {
                    float scale = 1.0f / range;
                    for (int i = 0; i < sequenceLength; i++) {
                        gradientOut[i] = gradientIn[i] * scale;
                    }
                } else {
                    // Zero gradient if no range
                    // gradientOut already initialized to zeros
                }
                break;
                
            case AUTO_NORMALIZE:
                // Gradient flows through normalization
                NormalizationStats currentStats = stats.get();
                double stdDev = Math.sqrt(currentStats.variance);
                if (stdDev > 1e-6) {
                    float scale = (float) (1.0 / stdDev);
                    for (int i = 0; i < sequenceLength; i++) {
                        gradientOut[i] = gradientIn[i] * scale;
                    }
                }
                break;
        }
        
        return gradientOut;
    }
    
    @Override
    public void applyGradients(float[][] weightGradients, float[] biasGradients) {
        // No weights to update in this layer
    }
    
    @Override
    public int getOutputSize() {
        return sequenceLength; // Flattened size is same as input
    }
    
    /**
     * Update running statistics for AUTO_NORMALIZE mode.
     * Uses Welford's online algorithm for numerical stability.
     */
    private void updateStatistics(float[] values) {
        if (stats == null) return;
        
        NormalizationStats newStats = new NormalizationStats();
        NormalizationStats oldStats = stats.get();
        
        // Initialize from old stats
        newStats.count = oldStats.count;
        newStats.mean = oldStats.mean;
        newStats.variance = oldStats.variance;
        
        // Update with new values using Welford's algorithm
        for (float value : values) {
            newStats.count++;
            double delta = value - newStats.mean;
            newStats.mean += delta / newStats.count;
            double delta2 = value - newStats.mean;
            newStats.variance += (delta * delta2 - newStats.variance) / newStats.count;
        }
        
        stats.set(newStats);
    }
    
    /**
     * Create a specification for sequence numerical input layers.
     */
    public static Layer.Spec spec(int sequenceLength, Feature feature, Optimizer optimizer) {
        return new SequenceNumericalSpec(sequenceLength, feature, optimizer);
    }
    
    /**
     * Specification for creating sequence numerical input layers.
     */
    static class SequenceNumericalSpec implements Layer.Spec {
        private final int sequenceLength;
        private final Feature feature;
        private final Optimizer optimizer;
        
        SequenceNumericalSpec(int sequenceLength, Feature feature, Optimizer optimizer) {
            this.sequenceLength = sequenceLength;
            this.feature = feature;
            this.optimizer = optimizer;
        }
        
        @Override
        public boolean prefersShapeAPI() {
            return true;
        }
        
        @Override
        public Layer create(int inputSize, Optimizer defaultOptimizer, FastRandom random) {
            if (inputSize != sequenceLength) {
                throw new IllegalArgumentException(
                    "Input size " + inputSize + " does not match sequence length " + sequenceLength);
            }
            Optimizer effectiveOptimizer = (optimizer != null) ? optimizer : defaultOptimizer;
            return new SequenceNumericalInputLayer(sequenceLength, feature, effectiveOptimizer);
        }
        
        @Override
        public Layer create(Shape inputShape, Optimizer defaultOptimizer, FastRandom random) {
            // We expect either a 1D shape matching sequence length, or no input shape
            if (inputShape != null && inputShape.toFlatSize() != sequenceLength) {
                throw new IllegalArgumentException(
                    "Input shape " + inputShape + " size does not match sequence length " + sequenceLength);
            }
            Optimizer effectiveOptimizer = (optimizer != null) ? optimizer : defaultOptimizer;
            return new SequenceNumericalInputLayer(sequenceLength, feature, effectiveOptimizer);
        }
        
        @Override
        public Shape getOutputShape(Shape inputShape) {
            return Shape.sequence(sequenceLength, 1);
        }
        
        @Override
        public void validateInputShape(Shape inputShape) {
            if (inputShape != null && inputShape.toFlatSize() != sequenceLength) {
                throw new IllegalArgumentException(
                    "Input shape " + inputShape + " (size=" + inputShape.toFlatSize() + 
                    ") does not match configured sequence length " + sequenceLength);
            }
        }
        
        @Override
        public int getOutputSize() {
            return sequenceLength;
        }
        
        @Override
        public int getOutputSize(int inputSize) {
            if (inputSize != sequenceLength) {
                throw new IllegalArgumentException(
                    "Input size " + inputSize + " does not match sequence length " + sequenceLength);
            }
            return sequenceLength;
        }
    }
}