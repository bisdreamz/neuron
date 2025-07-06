package dev.neuronic.net.layers;

import dev.neuronic.net.math.FastRandom;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.serialization.Serializable;
import dev.neuronic.net.serialization.SerializationConstants;
import dev.neuronic.net.serialization.SerializationService;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;

/**
 * Layer Normalization as a standalone layer.
 * 
 * <p><b>What it does:</b> Normalizes inputs across features to have zero mean and 
 * unit variance, then applies learnable scale (gamma) and shift (beta) parameters.
 * 
 * <p><b>Benefits:</b>
 * <ul>
 *   <li>Stabilizes training by reducing internal covariate shift</li>
 *   <li>Allows higher learning rates for faster convergence</li>
 *   <li>Especially beneficial for RNNs and small datasets</li>
 *   <li>Works with any batch size (even 1)</li>
 * </ul>
 * 
 * <p><b>Usage:</b>
 * <pre>{@code
 * NeuralNet model = NeuralNet.newBuilder()
 *     .input(sequenceLength)
 *     .layer(Layers.inputSequenceEmbedding(seqLen, vocabSize, embedDim))
 *     .layer(Layers.hiddenGruAll(hiddenSize))
 *     .layer(Layers.layerNorm())  // Normalize GRU outputs
 *     .layer(Layers.hiddenDenseRelu(hiddenSize))
 *     .output(Layers.outputSoftmaxCrossEntropy(vocabSize));
 * }</pre>
 */
public class LayerNormLayer implements Layer, Serializable {
    
    private final LayerNorm layerNorm;
    private final Optimizer optimizer;
    private final int size;
    
    // Thread-local buffers
    private final ThreadLocal<LayerNormContext> contextBuffer;
    private final ThreadLocal<float[]> gammaGradBuffer;
    private final ThreadLocal<float[]> betaGradBuffer;
    
    /**
     * Context for layer normalization that stores normalized values and stats.
     */
    public static class LayerNormContext extends LayerContext {
        public final float[] normalized;
        public final LayerNorm.Stats stats;
        
        public LayerNormContext(float[] inputs, float[] outputs, float[] normalized, LayerNorm.Stats stats) {
            super(inputs, null, outputs);
            this.normalized = normalized;
            this.stats = stats;
        }
        
    }
    
    /**
     * Create a layer normalization layer.
     * 
     * @param optimizer optimizer for gamma and beta parameters
     * @param size the feature dimension size
     * @param epsilon small constant for numerical stability
     */
    public LayerNormLayer(Optimizer optimizer, int size, float epsilon) {
        this.optimizer = optimizer;
        this.size = size;
        this.layerNorm = new LayerNorm(size, epsilon);
        
        // Thread-local buffers
        this.contextBuffer = ThreadLocal.withInitial(() -> 
            new LayerNormContext(new float[size], new float[size], new float[size], new LayerNorm.Stats()));
        this.gammaGradBuffer = ThreadLocal.withInitial(() -> new float[size]);
        this.betaGradBuffer = ThreadLocal.withInitial(() -> new float[size]);
    }
    
    /**
     * Create with default epsilon (1e-5).
     */
    public LayerNormLayer(Optimizer optimizer, int size) {
        this(optimizer, size, 1e-5f);
    }
    
    @Override
    public LayerContext forward(float[] input, boolean isTraining) {
        if (input.length != size) {
            throw new IllegalArgumentException("Input size mismatch: expected " + size + ", got " + input.length);
        }
        
        // Allocate new arrays for output - never return ThreadLocal buffers
        float[] outputs = new float[size];
        float[] normalized = new float[size];
        
        // Apply layer normalization
        LayerNorm.Stats stats = layerNorm.forward(input, outputs, normalized);
        
        // Create fresh stats object
        LayerNorm.Stats freshStats = new LayerNorm.Stats();
        freshStats.mean = stats.mean;
        freshStats.variance = stats.variance;
        
        return new LayerNormContext(input, outputs, normalized, freshStats);
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient) {
        LayerNormContext context = (LayerNormContext) stack[stackIndex];
        
        float[] gammaGrad = gammaGradBuffer.get();
        float[] betaGrad = betaGradBuffer.get();
        float[] inputGrad = new float[size];
        
        // Compute gradients
        layerNorm.backward(upstreamGradient, context.normalized, context.stats, 
                          inputGrad, gammaGrad, betaGrad);
        
        // Update parameters using optimizer
        // Note: We need to adapt optimizer interface for 1D parameters
        float[][] gammaGrad2D = {gammaGrad};
        float[][] gamma2D = new float[][]{layerNorm.getGamma()};
        
        optimizer.optimize(gamma2D, layerNorm.getBeta(), gammaGrad2D, betaGrad);
        
        return inputGrad;
    }
    
    @Override
    public int getOutputSize() {
        return size;
    }
    
    @Override
    public Optimizer getOptimizer() {
        return optimizer;
    }
    
    // Serialization
    
    @Override
    public void writeTo(DataOutputStream out, int version) throws IOException {
        out.writeInt(size);
        out.writeFloat(layerNorm.getEpsilon());
        
        // Write gamma and beta parameters
        float[] gamma = layerNorm.getGamma();
        float[] beta = layerNorm.getBeta();
        for (int i = 0; i < size; i++) {
            out.writeFloat(gamma[i]);
        }
        for (int i = 0; i < size; i++) {
            out.writeFloat(beta[i]);
        }
        
        // Write optimizer
        Serializable serializableOptimizer = (Serializable) optimizer;
        out.writeInt(serializableOptimizer.getTypeId());
        serializableOptimizer.writeTo(out, version);
    }
    
    @Override
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use deserialize() static method instead");
    }
    
    public static LayerNormLayer deserialize(DataInputStream in, int version, FastRandom random) throws IOException {
        int size = in.readInt();
        float epsilon = in.readFloat();
        
        // Read gamma and beta
        float[] gamma = new float[size];
        float[] beta = new float[size];
        for (int i = 0; i < size; i++) {
            gamma[i] = in.readFloat();
        }
        for (int i = 0; i < size; i++) {
            beta[i] = in.readFloat();
        }
        
        // Read optimizer
        int optimizerTypeId = in.readInt();
        Optimizer optimizer = SerializationService.deserializeOptimizer(in, optimizerTypeId, version);
        
        // Create layer and restore parameters
        LayerNormLayer layer = new LayerNormLayer(optimizer, size, epsilon);
        System.arraycopy(gamma, 0, layer.layerNorm.getGamma(), 0, size);
        System.arraycopy(beta, 0, layer.layerNorm.getBeta(), 0, size);
        
        return layer;
    }
    
    @Override
    public int getSerializedSize(int version) {
        int size = 8; // size + epsilon
        size += this.size * 8; // gamma + beta
        size += 4; // optimizer type ID
        size += ((Serializable) optimizer).getSerializedSize(version);
        return size;
    }
    
    @Override
    public int getTypeId() {
        return SerializationConstants.TYPE_LAYER_NORM_LAYER;
    }
    
    /**
     * Create a layer normalization layer specification.
     */
    public static Layer.Spec spec(Optimizer optimizer, float epsilon) {
        return new LayerNormSpec(optimizer, epsilon);
    }
    
    /**
     * Create with default epsilon.
     */
    public static Layer.Spec spec(Optimizer optimizer) {
        return spec(optimizer, 1e-5f);
    }
    
    /**
     * Create with default optimizer and epsilon.
     */
    public static Layer.Spec spec() {
        return spec(null, 1e-5f);
    }
    
    /**
     * Specification for layer normalization layers.
     */
    private static class LayerNormSpec implements Layer.Spec {
        private final Optimizer optimizer;
        private final float epsilon;
        
        public LayerNormSpec(Optimizer optimizer, float epsilon) {
            this.optimizer = optimizer;
            this.epsilon = epsilon;
        }
        
        @Override
        public int getOutputSize() {
            return -1; // Output size matches input size
        }
        
        @Override
        public int getOutputSize(int inputSize) {
            return inputSize;
        }
        
        
        public Layer create(int inputSize, Optimizer defaultOptimizer) {
            Optimizer effectiveOptimizer = (optimizer != null) ? optimizer : defaultOptimizer;
            if (effectiveOptimizer == null) {
                throw new IllegalStateException("No optimizer available for LayerNorm parameters");
            }
            return new LayerNormLayer(effectiveOptimizer, inputSize, epsilon);
        }
        
        public Layer create(int inputSize, Optimizer defaultOptimizer, FastRandom random) {
            // LayerNorm doesn't use random initialization
            return create(inputSize, defaultOptimizer);
        }
    }
}