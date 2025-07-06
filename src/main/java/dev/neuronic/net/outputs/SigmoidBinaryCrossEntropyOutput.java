package dev.neuronic.net.outputs;

import dev.neuronic.net.layers.Layer;
import dev.neuronic.net.layers.BaseLayerSpec;
import dev.neuronic.net.common.PooledFloatArray;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.math.NetMath;
import dev.neuronic.net.serialization.Serializable;
import dev.neuronic.net.serialization.SerializationConstants;
import dev.neuronic.net.serialization.SerializationService;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Sigmoid + Binary Cross-Entropy output for binary classification.
 * 
 * Computes: sigmoid(Wx + b) for probability
 * Uses Binary Cross-Entropy loss internally.
 * 
 * Use for:
 * - Binary classification (spam/not spam, dog/cat)
 * - Single probability output (0.0 to 1.0)
 * 
 * Output interpretation:
 * - > 0.5: Positive class
 * - < 0.5: Negative class
 */
public class SigmoidBinaryCrossEntropyOutput implements Layer, Serializable, RegressionOutput {
    
    private final Optimizer optimizer;
    private final float[][] weights;
    private final float[] biases;
    private final int inputs;
    // Instance buffer pools for different array sizes
    private final PooledFloatArray inputBufferPool;       // For input-sized arrays
    
    public SigmoidBinaryCrossEntropyOutput(Optimizer optimizer, int inputs) {
        this.optimizer = optimizer;
        this.weights = new float[inputs][1]; // Single output
        this.biases = new float[1];
        this.inputs = inputs;
        // Initialize buffer pools
        this.inputBufferPool = new PooledFloatArray(inputs);
        
        // Xavier initialization for sigmoid
        NetMath.weightInitXavier(weights, inputs, 1);
        NetMath.biasInit(biases, 0.0f);
    }
    
    @Override
    public LayerContext forward(float[] input, boolean isTraining) {
        // Allocate new arrays for LayerContext - never use ThreadLocal buffers in contexts
        float[] logit = new float[1];
        logit[0] = biases[0];
        for (int i = 0; i < inputs; i++) {
            logit[0] += input[i] * weights[i][0];
        }
        
        float[] probability = new float[1];
        probability[0] = 1.0f / (1.0f + (float) Math.exp(-logit[0]));
        
        return new LayerContext(input, logit, probability);
    }
    
    /**
     * Compute Binary Cross-Entropy loss.
     * 
     * @param probability Single probability value [0, 1]
     * @param target Single target value (0.0 or 1.0)
     */
    public float computeLoss(float probability, float target) {
        // Clip probability to prevent log(0)
        float clipped = Math.max(Math.min(probability, 0.9999999f), 0.0000001f);
        
        return -(target * (float) Math.log(clipped) + (1 - target) * (float) Math.log(1 - clipped));
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] targets) {
        LayerContext context = stack[stackIndex];
        
        float[] downstreamGradient = inputBufferPool.getBuffer();
        
        try {
            // Binary cross-entropy + sigmoid gradient: prediction - target
            float probability = context.outputs()[0];
            float target = targets[0];
            float gradient = probability - target;
            
            // Compute weight gradients
            float[][] weightGradients = new float[inputs][1];
            for (int i = 0; i < inputs; i++) {
                weightGradients[i][0] = context.inputs()[i] * gradient;
            }
            
            // Update weights and biases
            optimizer.optimize(weights, biases, weightGradients, new float[]{gradient});
            
            // Compute downstream gradient
            for (int i = 0; i < inputs; i++) {
                downstreamGradient[i] = gradient * weights[i][0];
            }
            
            // Return a fresh copy
            float[] result = new float[inputs];
            System.arraycopy(downstreamGradient, 0, result, 0, inputs);
            return result;
            
        } finally {
            inputBufferPool.releaseBuffer(downstreamGradient);
        }
    }
    
    @Override
    public int getOutputSize() {
        return 1;
    }
    
    public static Layer.Spec spec(Optimizer optimizer) {
        return new SigmoidBinaryCrossEntropyOutputSpec(optimizer, 1.0);
    }
    
    /**
     * Create a sigmoid binary cross-entropy output specification with custom learning rate ratio.
     * 
     * @param optimizer optimizer for this layer (null to use default)
     * @param learningRateRatio learning rate scaling factor (1.0 = normal)
     */
    public static Layer.Spec spec(Optimizer optimizer, double learningRateRatio) {
        return new SigmoidBinaryCrossEntropyOutputSpec(optimizer, learningRateRatio);
    }
    
    /**
     * Specification for creating sigmoid binary cross-entropy output layers with optimizer management.
     */
    private static class SigmoidBinaryCrossEntropyOutputSpec extends BaseLayerSpec<SigmoidBinaryCrossEntropyOutputSpec> {
        
        public SigmoidBinaryCrossEntropyOutputSpec(Optimizer optimizer, double learningRateRatio) {
            super(1, optimizer);
            this.learningRateRatio = (float) learningRateRatio;
        }
        
        @Override
        public Layer create(int inputSize) {
            return createLayer(inputSize, getEffectiveOptimizer(null));
        }
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer) {
            return new SigmoidBinaryCrossEntropyOutput(effectiveOptimizer, inputSize);
        }
        
    }
    
    // Serialization implementation
    
    @Override
    public void writeTo(DataOutputStream out, int version) throws IOException {
        // Write layer dimensions (always 1 output for binary classification)
        out.writeInt(1);
        out.writeInt(inputs);
        
        // Write weights (column-major format)
        for (int i = 0; i < inputs; i++) {
            out.writeFloat(weights[i][0]);
        }
        
        // Write bias
        out.writeFloat(biases[0]);
        
        // Write optimizer
        Serializable serializableOptimizer = (Serializable) optimizer;
        out.writeInt(serializableOptimizer.getTypeId());
        serializableOptimizer.writeTo(out, version);
    }
    
    @Override
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use deserialize(DataInputStream, int) static method instead");
    }
    
    /**
     * Static method to deserialize a SigmoidBinaryCrossEntropyOutput from stream.
     */
    public static SigmoidBinaryCrossEntropyOutput deserialize(DataInputStream in, int version) throws IOException {
        // Read layer dimensions
        int outputs = in.readInt(); // Should be 1 for binary classification
        int inputs = in.readInt();
        
        // Read weights
        float[][] weights = new float[inputs][1];
        for (int i = 0; i < inputs; i++) {
            weights[i][0] = in.readFloat();
        }
        
        // Read bias
        float bias = in.readFloat();
        
        // Read optimizer using centralized service
        int optimizerTypeId = in.readInt();
        Optimizer optimizer = SerializationService.deserializeOptimizer(in, optimizerTypeId, version);
        
        // Create layer and set weights
        SigmoidBinaryCrossEntropyOutput layer = new SigmoidBinaryCrossEntropyOutput(optimizer, inputs);
        
        // Copy weights and bias
        for (int i = 0; i < inputs; i++) {
            layer.weights[i][0] = weights[i][0];
        }
        layer.biases[0] = bias;
        
        return layer;
    }
    
    @Override
    public int getSerializedSize(int version) {
        int size = 8; // outputs + inputs
        size += inputs * 4; // weights (1 output neuron)
        size += 4; // bias
        size += 4; // optimizer type ID
        size += ((Serializable) optimizer).getSerializedSize(version); // optimizer data
        return size;
    }
    
    @Override
    public int getTypeId() {
        return SerializationConstants.TYPE_SIGMOID_BINARY_OUTPUT;
    }
}