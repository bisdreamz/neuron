package dev.neuronic.net.outputs;

import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.layers.BaseLayerSpec;
import dev.neuronic.net.common.PooledFloatArray;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.math.FastRandom;
import dev.neuronic.net.math.NetMath;

/**
 * Multi-label sigmoid output for independent binary classifications.
 * 
 * Each output is an independent binary classifier with sigmoid activation.
 * Uses Binary Cross-Entropy loss for each output independently.
 * 
 * Use for:
 * - Multi-label classification (tags, categories that aren't mutually exclusive)
 * - Multiple independent yes/no decisions
 * - Document classification with multiple topics
 * 
 * Example: Image can be tagged as both "outdoor" AND "sunny" AND "people"
 */
public class MultiLabelSigmoidOutput implements Layer {
    
    private final Optimizer optimizer;
    private final float[][] weights;
    private final float[] biases;
    private final int labels;
    private final int inputs;
    // Instance buffer pools for different array sizes
    private final PooledFloatArray labelBufferPool;       // For label-sized arrays
    private final PooledFloatArray inputBufferPool;       // For input-sized arrays
    
    public MultiLabelSigmoidOutput(Optimizer optimizer, int labels, int inputs, FastRandom random) {
        this.optimizer = optimizer;
        this.weights = new float[inputs][labels];
        this.biases = new float[labels];
        this.labels = labels;
        this.inputs = inputs;
        // Initialize buffer pools
        this.labelBufferPool = new PooledFloatArray(labels);
        this.inputBufferPool = new PooledFloatArray(inputs);
        
        // Xavier initialization for sigmoid
        NetMath.weightInitXavier(weights, inputs, labels, random);
        NetMath.biasInit(biases, 0.0f);
    }
    
    @Override
    public LayerContext forward(float[] input, boolean isTraining) {
        // Compute logits using ThreadLocal buffer
        // Allocate new arrays for LayerContext - never use ThreadLocal buffers in contexts
        float[] logits = new float[labels];
        System.arraycopy(biases, 0, logits, 0, labels);
        
        for (int inputIdx = 0; inputIdx < inputs; inputIdx++) {
            float inputValue = input[inputIdx];
            for (int labelIdx = 0; labelIdx < labels; labelIdx++) {
                logits[labelIdx] += inputValue * weights[inputIdx][labelIdx];
            }
        }
        
        // Apply sigmoid to each output independently
        float[] probabilities = new float[labels];
        for (int i = 0; i < labels; i++) {
            probabilities[i] = 1.0f / (1.0f + (float) Math.exp(-logits[i]));
        }
        
        return new LayerContext(input, logits, probabilities);
    }
    
    /**
     * Compute total Binary Cross-Entropy loss across all labels.
     */
    public float computeLoss(float[] probabilities, float[] targets) {
        float totalLoss = 0;
        
        for (int i = 0; i < labels; i++) {
            // Clip probability to prevent log(0)
            float clipped = Math.max(Math.min(probabilities[i], 0.9999999f), 0.0000001f);
            
            float loss = -(targets[i] * (float) Math.log(clipped) + 
                          (1 - targets[i]) * (float) Math.log(1 - clipped));
            totalLoss += loss;
        }
        
        return totalLoss / labels; // Average loss per label
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] targets) {
        LayerContext context = stack[stackIndex];
        
        float[] gradients = labelBufferPool.getBuffer();
        float[] downstreamGradient = inputBufferPool.getBuffer();
        
        try {
            // Binary cross-entropy + sigmoid gradient for each label: prediction - target
            for (int i = 0; i < labels; i++) {
                gradients[i] = context.outputs()[i] - targets[i];
            }
            
            // Compute weight gradients
            float[][] weightGradients = new float[inputs][labels];
            for (int inputIdx = 0; inputIdx < inputs; inputIdx++) {
                for (int labelIdx = 0; labelIdx < labels; labelIdx++) {
                    weightGradients[inputIdx][labelIdx] = context.inputs()[inputIdx] * gradients[labelIdx];
                }
            }
            
            // Update weights and biases
            optimizer.optimize(weights, biases, weightGradients, gradients);
            
            // Compute downstream gradient
            for (int inputIdx = 0; inputIdx < inputs; inputIdx++) {
                float sum = 0;
                for (int labelIdx = 0; labelIdx < labels; labelIdx++) {
                    sum += gradients[labelIdx] * weights[inputIdx][labelIdx];
                }
                downstreamGradient[inputIdx] = sum;
            }
            
            // Return a fresh copy
            float[] result = new float[inputs];
            System.arraycopy(downstreamGradient, 0, result, 0, inputs);
            return result;
            
        } finally {
            labelBufferPool.releaseBuffer(gradients);
            inputBufferPool.releaseBuffer(downstreamGradient);
        }
    }
    
    @Override
    public int getOutputSize() {
        return labels;
    }
    
    public static Layer.Spec spec(int labels, Optimizer optimizer) {
        return new MultiLabelSigmoidOutputSpec(labels, optimizer, 1.0);
    }
    
    /**
     * Create a multi-label sigmoid output specification with custom learning rate ratio.
     * 
     * @param labels number of independent labels
     * @param optimizer optimizer for this layer (null to use default)
     * @param learningRateRatio learning rate scaling factor (1.0 = normal)
     */
    public static Layer.Spec spec(int labels, Optimizer optimizer, double learningRateRatio) {
        return new MultiLabelSigmoidOutputSpec(labels, optimizer, learningRateRatio);
    }
    
    /**
     * Specification for creating multi-label sigmoid output layers with optimizer management.
     */
    private static class MultiLabelSigmoidOutputSpec extends BaseLayerSpec<MultiLabelSigmoidOutputSpec> {
        private final int labels;
        
        public MultiLabelSigmoidOutputSpec(int labels, Optimizer optimizer, double learningRateRatio) {
            super(labels, optimizer);
            this.labels = labels;
            this.learningRateRatio = (float) learningRateRatio;
        }
        
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer, FastRandom random) {
            return new MultiLabelSigmoidOutput(effectiveOptimizer, labels, inputSize, random);
        }
        
        @Override
        public int getOutputSize() {
            return labels;
        }
    }
}