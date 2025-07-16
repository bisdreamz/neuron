package dev.neuronic.net.outputs;

import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.layers.GradientAccumulator;
import dev.neuronic.net.layers.BaseLayerSpec;
import dev.neuronic.net.common.PooledFloatArray;
import dev.neuronic.net.activators.SoftmaxActivator;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.math.FastRandom;
import dev.neuronic.net.math.NetMath;
import dev.neuronic.net.serialization.Serializable;
import dev.neuronic.net.serialization.SerializationConstants;
import dev.neuronic.net.serialization.SerializationService;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Fused Softmax activation + Cross-Entropy loss output layer.
 * 
 * This is the standard, mathematically optimal way to handle multi-class classification.
 * The gradient computation is simplified to: softmax(logits) - targets
 * 
 * Benefits:
 * - Numerically stable (works with logits directly)
 * - Mathematically correct (no double-derivative issues)
 * - No configuration errors possible
 * - Standard in all major ML frameworks
 */
public class SoftmaxCrossEntropyOutput implements Layer, GradientAccumulator, Serializable {
    
    private final Optimizer optimizer;
    private final float[][] weights; // Column-major: weights[input][neuron] 
    private final float[] biases;
    private final int neurons;
    private final int inputs;
    // Instance buffer pools for different array sizes
    private final PooledFloatArray neuronBufferPool;      // For neuron-sized arrays
    private final PooledFloatArray inputBufferPool;       // For input-sized arrays
    
    // Gradient accumulation state
    private final ThreadLocal<float[][]> accumulatedWeightGradients;
    private final ThreadLocal<float[]> accumulatedBiasGradients;
    private final ThreadLocal<Boolean> accumulating;
    
    public SoftmaxCrossEntropyOutput(Optimizer optimizer, int neurons, int inputs, WeightInitStrategy initStrategy, FastRandom random) {
        this.optimizer = optimizer;
        this.weights = new float[inputs][neurons];
        this.biases = new float[neurons];
        this.neurons = neurons;
        this.inputs = inputs;
        // Initialize buffer pools
        this.neuronBufferPool = new PooledFloatArray(neurons);
        this.inputBufferPool = new PooledFloatArray(inputs);
        
        // Initialize gradient accumulation state
        this.accumulatedWeightGradients = ThreadLocal.withInitial(() -> new float[inputs][neurons]);
        this.accumulatedBiasGradients = ThreadLocal.withInitial(() -> new float[neurons]);
        this.accumulating = ThreadLocal.withInitial(() -> Boolean.FALSE);
        
        // Initialize weights and biases
        switch (initStrategy) {
            case XAVIER -> NetMath.weightInitXavier(weights, inputs, neurons, random);
            case HE -> NetMath.weightInitHe(weights, inputs, random);
        }
        NetMath.biasInit(biases, 0.0f);
    }
    
    @Override
    public LayerContext forward(float[] input, boolean isTraining) {
        // Allocate new arrays for LayerContext - never use ThreadLocal buffers in contexts
        float[] logits = new float[neurons];
        NetMath.matrixPreActivationsColumnMajor(input, weights, biases, logits);
        
        // Apply softmax to get probabilities
        float[] probabilities = new float[neurons];
        SoftmaxActivator.INSTANCE.activate(logits, probabilities);
        
        return new LayerContext(input, logits, probabilities);
    }
    
    
    /**
     * Compute loss for this fused output layer.
     */
    public float computeLoss(float[] probabilities, float[] targets) {
        // Check if this is a sparse target
        if (targets.length == 1) {
            int targetIndex = (int) targets[0];
            if (targetIndex < 0 || targetIndex >= probabilities.length) {
                throw new IllegalArgumentException(
                    "Target index " + targetIndex + " out of bounds [0, " + probabilities.length + ")");
            }
            // Cross-entropy = -log(p[target])
            return (float) -Math.log(probabilities[targetIndex] + 1e-7);
        } else {
            // Dense targets
            return NetMath.lossComputeCrossEntropy(targets, probabilities);
        }
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] targets) {
        LayerContext context = stack[stackIndex];
        
        float[] gradients = neuronBufferPool.getBuffer();
        float[] downstreamGradient = inputBufferPool.getBuffer();
        
        try {
            // Check if this is a sparse target (single element with class index)
            if (targets.length == 1) {
                // Sparse cross-entropy: gradient = softmax_output with target class -= 1
                int targetIndex = (int) targets[0];
                if (targetIndex < 0 || targetIndex >= neurons) {
                    throw new IllegalArgumentException(
                        "Target index " + targetIndex + " out of bounds [0, " + neurons + ")");
                }
                System.arraycopy(context.outputs(), 0, gradients, 0, neurons);
                gradients[targetIndex] -= 1.0f;
            } else {
                // Dense cross-entropy: gradient = softmax_output - one_hot_targets
                NetMath.elementwiseSubtract(context.outputs(), targets, gradients);
            }
            
            // Compute weight gradients using NetMath
            float[][] weightGradients = new float[inputs][neurons];
            NetMath.matrixWeightGradientsColumnMajor(context.inputs(), gradients, weightGradients);
            
            // Update weights and biases
            optimizer.optimize(weights, biases, weightGradients, gradients);
            
            // Compute downstream gradient using NetMath
            NetMath.matrixVectorMultiplyColumnMajor(weights, gradients, downstreamGradient);
            
            // Return a fresh copy
            float[] result = new float[inputs];
            System.arraycopy(downstreamGradient, 0, result, 0, inputs);
            return result;
            
        } finally {
            neuronBufferPool.releaseBuffer(gradients);
            inputBufferPool.releaseBuffer(downstreamGradient);
        }
    }
    
    @Override
    public int getOutputSize() {
        return neurons;
    }
    
    @Override
    public Optimizer getOptimizer() {
        return optimizer;
    }
    
    @Override
    public float[] computeGradientWithTargets(LayerContext[] stack, int stackIndex,
                                              float[] targets, GradientConsumer gradientConsumer) {
        LayerContext context = stack[stackIndex];
        
        // This is the fused gradient for Softmax + Cross-Entropy.
        float[] predictions = context.outputs();
        float[] gradient = new float[predictions.length];

        if (targets.length == 1) {
            // Sparse target (class index)
            int trueClassIndex = (int) targets[0];
            System.arraycopy(predictions, 0, gradient, 0, predictions.length);
            if (trueClassIndex >= 0 && trueClassIndex < gradient.length) {
                gradient[trueClassIndex] -= 1.0f;
            }
        } else {
            // Dense target (one-hot encoded)
            NetMath.elementwiseSubtract(predictions, targets, gradient);
        }

        // Compute weight gradients for this layer
        float[][] weightGradients = new float[inputs][neurons];
        NetMath.matrixWeightGradientsColumnMajor(context.inputs(), gradient, weightGradients);
        
        // Pass gradients to consumer
        if (gradientConsumer != null) {
            gradientConsumer.accept(stackIndex, weightGradients, gradient);
        }
        
        // Compute and return downstream gradient
        float[] downstreamGradient = new float[inputs];
        NetMath.matrixVectorMultiplyColumnMajor(weights, gradient, downstreamGradient);
        
        return downstreamGradient;
    }
    
    @Override
    public GradientDimensions getGradientDimensions() {
        return new GradientDimensions(inputs, neurons, neurons);
    }
    
    @Override
    public void applyGradients(float[][] weightGradients, float[] biasGradients) {
        optimizer.optimize(weights, biases, weightGradients, biasGradients);
    }
    
    // Gradient accumulation implementation
    
    @Override
    public void startAccumulation() {
        accumulating.set(Boolean.TRUE);
        // Zero out accumulated gradients
        float[][] weightGrads = accumulatedWeightGradients.get();
        float[] biasGrads = accumulatedBiasGradients.get();
        
        NetMath.matrixInit(weightGrads, 0.0f);
        NetMath.biasInit(biasGrads, 0.0f);
    }
    
    @Override
    public float[] backwardAccumulate(LayerContext[] stack, int stackIndex, float[] targets) {
        LayerContext context = stack[stackIndex];
        
        float[] gradients = neuronBufferPool.getBuffer();
        
        try {
            // Check if this is a sparse target
            if (targets.length == 1) {
                // Sparse cross-entropy gradient
                int targetIndex = (int) targets[0];
                if (targetIndex < 0 || targetIndex >= neurons) {
                    throw new IllegalArgumentException(
                        "Target index " + targetIndex + " out of bounds [0, " + neurons + ")");
                }
                System.arraycopy(context.outputs(), 0, gradients, 0, neurons);
                gradients[targetIndex] -= 1.0f;
            } else {
                // Dense cross-entropy gradient
                NetMath.elementwiseSubtract(context.outputs(), targets, gradients);
            }
            
            // Compute weight gradients using NetMath
            float[][] weightGradients = new float[inputs][neurons];
            NetMath.matrixWeightGradientsColumnMajor(context.inputs(), gradients, weightGradients);
            
            // Accumulate gradients
            float[][] accWeightGrads = accumulatedWeightGradients.get();
            float[] accBiasGrads = accumulatedBiasGradients.get();
            
            // Add weight gradients to accumulated
            for (int i = 0; i < inputs; i++) {
                NetMath.elementwiseAdd(accWeightGrads[i], weightGradients[i], accWeightGrads[i]);
            }
            
            // Add bias gradients to accumulated
            NetMath.elementwiseAdd(accBiasGrads, gradients, accBiasGrads);
            
            // Compute downstream gradient using NetMath
            float[] downstreamGradient = inputBufferPool.getBuffer();
            try {
                NetMath.matrixVectorMultiplyColumnMajor(weights, gradients, downstreamGradient);
                
                // Return a fresh copy
                float[] result = new float[inputs];
                System.arraycopy(downstreamGradient, 0, result, 0, inputs);
                return result;
            } finally {
                inputBufferPool.releaseBuffer(downstreamGradient);
            }
            
        } finally {
            neuronBufferPool.releaseBuffer(gradients);
        }
    }
    
    @Override
    public void applyAccumulatedGradients(int batchSize) {
        if (!accumulating.get()) return;
        
        float[][] accWeightGrads = accumulatedWeightGradients.get();
        float[] accBiasGrads = accumulatedBiasGradients.get();
        
        // Average the accumulated gradients
        float scale = 1.0f / batchSize;
        for (int i = 0; i < inputs; i++) {
            NetMath.elementwiseScaleInPlace(accWeightGrads[i], scale);
        }
        NetMath.elementwiseScaleInPlace(accBiasGrads, scale);
        
        // Update weights and biases
        optimizer.optimize(weights, biases, accWeightGrads, accBiasGrads);
        
        accumulating.set(Boolean.FALSE);
    }
    
    @Override
    public boolean isAccumulating() {
        return accumulating.get();
    }
    
    /**
     * Create a specification for a Softmax + Cross-Entropy output layer.
     * This is the recommended way to handle multi-class classification.
     */
    public static Layer.Spec spec(int classes, Optimizer optimizer, WeightInitStrategy initStrategy) {
        return new SoftmaxCrossEntropyOutputSpec(classes, optimizer, initStrategy, 1.0);
    }
    
    /**
     * Create a softmax cross-entropy output specification with custom learning rate ratio.
     * 
     * @param classes number of output classes
     * @param optimizer optimizer for this layer (null to use default)
     * @param initStrategy weight initialization strategy
     * @param learningRateRatio learning rate scaling factor (1.0 = normal)
     */
    public static Layer.Spec spec(int classes, Optimizer optimizer, WeightInitStrategy initStrategy, double learningRateRatio) {
        return new SoftmaxCrossEntropyOutputSpec(classes, optimizer, initStrategy, learningRateRatio);
    }
    
    /**
     * Specification for creating softmax cross-entropy output layers with optimizer management.
     */
    private static class SoftmaxCrossEntropyOutputSpec extends BaseLayerSpec<SoftmaxCrossEntropyOutputSpec> {
        private final int classes;
        private final WeightInitStrategy initStrategy;
        
        public SoftmaxCrossEntropyOutputSpec(int classes, Optimizer optimizer, WeightInitStrategy initStrategy, double learningRateRatio) {
            super(classes, optimizer);
            this.classes = classes;
            this.initStrategy = initStrategy;
            this.learningRateRatio = (float) learningRateRatio;
        }
        
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer, FastRandom random) {
            return new SoftmaxCrossEntropyOutput(effectiveOptimizer, classes, inputSize, initStrategy, random);
        }
    }
    
    // Serialization implementation
    
    @Override
    public void writeTo(DataOutputStream out, int version) throws IOException {
        // Write layer dimensions
        out.writeInt(neurons);
        out.writeInt(inputs);
        
        // Write weights (column-major format)
        for (int i = 0; i < inputs; i++) {
            for (int j = 0; j < neurons; j++) {
                out.writeFloat(weights[i][j]);
            }
        }
        
        // Write biases
        for (float bias : biases) {
            out.writeFloat(bias);
        }
        
        // Write optimizer (assuming it's serializable - will need to implement this)
        Serializable serializableOptimizer = (Serializable) optimizer;
        out.writeInt(serializableOptimizer.getTypeId());
        serializableOptimizer.writeTo(out, version);
    }
    
    @Override
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use readFrom(DataInputStream, int) static method instead");
    }
    
    /**
     * Static method to deserialize a SoftmaxCrossEntropyOutput from stream.
     */
    public static SoftmaxCrossEntropyOutput deserialize(DataInputStream in, int version, FastRandom random) throws IOException {
        // Read layer dimensions
        int neurons = in.readInt();
        int inputs = in.readInt();
        
        // Read weights
        float[][] weights = new float[inputs][neurons];
        for (int i = 0; i < inputs; i++) {
            for (int j = 0; j < neurons; j++) {
                weights[i][j] = in.readFloat();
            }
        }
        
        // Read biases
        float[] biases = new float[neurons];
        for (int i = 0; i < neurons; i++) {
            biases[i] = in.readFloat();
        }
        
        // Read optimizer using centralized service
        int optimizerTypeId = in.readInt();
        Optimizer optimizer = SerializationService.deserializeOptimizer(in, optimizerTypeId, version);
        
        // Create layer with provided FastRandom
        SoftmaxCrossEntropyOutput layer = new SoftmaxCrossEntropyOutput(optimizer, neurons, inputs, WeightInitStrategy.XAVIER, random);
        
        // Overwrite with saved weights and biases
        System.arraycopy(biases, 0, layer.biases, 0, neurons);
        for (int i = 0; i < inputs; i++) {
            System.arraycopy(weights[i], 0, layer.weights[i], 0, neurons);
        }
        
        return layer;
    }
    
    @Override
    public int getSerializedSize(int version) {
        int size = 8; // neurons + inputs
        size += inputs * neurons * 4; // weights
        size += neurons * 4; // biases
        size += 4; // optimizer type ID
        size += ((Serializable) optimizer).getSerializedSize(version); // optimizer data
        return size;
    }
    
    @Override
    public int getTypeId() {
        return SerializationConstants.TYPE_SOFTMAX_CROSSENTROPY_OUTPUT;
    }
}