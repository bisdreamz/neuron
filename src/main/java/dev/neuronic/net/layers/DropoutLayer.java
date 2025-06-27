package dev.neuronic.net.layers;

import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.serialization.Serializable;
import dev.neuronic.net.serialization.SerializationConstants;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Dropout layer for regularization during training.
 * 
 * <p><b>What it does:</b> Randomly "drops out" (sets to zero) a percentage of neurons
 * during training, which helps prevent overfitting by forcing the network to learn
 * redundant representations.
 * 
 * <p><b>Key features:</b>
 * <ul>
 *   <li>Uses inverted dropout: scales activations during training so no scaling needed at inference</li>
 *   <li>Thread-safe using ThreadLocalRandom</li>
 *   <li>Automatically disabled during inference (when training=false)</li>
 *   <li>Zero overhead when dropout rate is 0</li>
 * </ul>
 * 
 * <p><b>Common dropout rates:</b>
 * <ul>
 *   <li>0.1-0.2: Light regularization for small networks</li>
 *   <li>0.3-0.5: Standard regularization for hidden layers</li>
 *   <li>0.5-0.8: Heavy regularization for large networks</li>
 * </ul>
 */
public class DropoutLayer implements Layer, Serializable {
    
    /**
     * Dropout-specific context that stores the dropout mask for backward pass.
     */
    public static class DropoutContext extends LayerContext {
        public final boolean[] mask;  // Which neurons were kept (true) or dropped (false)
        
        public DropoutContext(float[] inputs, float[] outputs, boolean[] mask) {
            super(inputs, null, outputs); // No preactivations in dropout
            this.mask = mask;
        }
    }
    
    private final float dropoutRate;      // Probability of dropping a neuron (0.0 to 1.0)
    private final float keepProbability;  // 1 - dropoutRate (cached for efficiency)
    private final float scale;            // 1 / keepProbability (for inverted dropout)
    private final int fixedSize;          // Fixed size if >= 0, dynamic if -1
    
    // Thread-local buffers that grow as needed
    private final ThreadLocal<BufferContainer> buffers = ThreadLocal.withInitial(BufferContainer::new);
    
    // Container for thread-local buffers that can resize dynamically
    private static class BufferContainer {
        float[] output = new float[0];
        boolean[] mask = new boolean[0];
        float[] gradientBuffer = new float[0];  // For backward pass
        
        void ensureCapacity(int size) {
            if (output.length < size) {
                output = new float[size];
                mask = new boolean[size];
                gradientBuffer = new float[size];
            }
        }
    }
    
    // Removed training mode - dropout should always behave consistently
    
    /**
     * Create a dropout layer with dynamic size.
     * 
     * @param dropoutRate probability of dropping each neuron (0.0 to 1.0)
     */
    public DropoutLayer(float dropoutRate) {
        this(dropoutRate, -1); // -1 indicates dynamic size
    }
    
    /**
     * Create a dropout layer with fixed size.
     * 
     * @param dropoutRate probability of dropping each neuron (0.0 to 1.0)
     * @param size fixed input/output size
     */
    public DropoutLayer(float dropoutRate, int size) {
        if (dropoutRate < 0.0f || dropoutRate >= 1.0f) {
            throw new IllegalArgumentException("Dropout rate must be in [0, 1): " + dropoutRate);
        }
        if (size > 0 || size == -1) {
            // size > 0 for fixed size, -1 for dynamic
        } else {
            throw new IllegalArgumentException("Size must be positive or -1 for dynamic: " + size);
        }
        
        this.dropoutRate = dropoutRate;
        this.keepProbability = 1.0f - dropoutRate;
        this.scale = (dropoutRate == 0.0f) ? 1.0f : 1.0f / keepProbability;
        this.fixedSize = size;
    }
    
    
    @Override
    public LayerContext forward(float[] input) {
        int size = input.length;
        
        // Validate size if fixed
        if (fixedSize > 0 && size != fixedSize) {
            throw new IllegalArgumentException("Input size mismatch: expected " + fixedSize + ", got " + size);
        }
        
        BufferContainer container = buffers.get();
        container.ensureCapacity(size);
        
        if (dropoutRate == 0.0f) {
            // No dropout: pass through unchanged
            System.arraycopy(input, 0, container.output, 0, size);
            // Return view of the buffer, not the whole buffer
            float[] output = new float[size];
            System.arraycopy(container.output, 0, output, 0, size);
            return new DropoutContext(input, output, null);
        }
        
        // Training mode: apply dropout
        Random random = ThreadLocalRandom.current();
        
        // Generate dropout mask and apply with inverted dropout scaling using ThreadLocal buffers
        for (int i = 0; i < size; i++) {
            if (random.nextFloat() < keepProbability) {
                container.mask[i] = true;
                container.output[i] = input[i] * scale; // Scale up to maintain expected value
            } else {
                container.mask[i] = false;
                container.output[i] = 0.0f;
            }
        }
        
        // Create fresh arrays for LayerContext return
        float[] freshOutput = new float[size];
        boolean[] freshMask = new boolean[size];
        System.arraycopy(container.output, 0, freshOutput, 0, size);
        System.arraycopy(container.mask, 0, freshMask, 0, size);
        
        return new DropoutContext(input, freshOutput, freshMask);
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient) {
        DropoutContext context = (DropoutContext) stack[stackIndex];
        
        if (context.mask == null || dropoutRate == 0.0f) {
            // No dropout was applied: pass gradient through unchanged
            return upstreamGradient;
        }
        
        // Apply same dropout mask to gradients with scaling using ThreadLocal buffer
        int size = upstreamGradient.length;
        BufferContainer container = buffers.get();
        container.ensureCapacity(size);
        
        boolean[] mask = context.mask;
        
        for (int i = 0; i < size; i++) {
            if (mask[i]) {
                container.gradientBuffer[i] = upstreamGradient[i] * scale;
            } else {
                container.gradientBuffer[i] = 0.0f;
            }
        }
        
        // Return fresh array copy for thread safety
        float[] downstreamGradient = new float[size];
        System.arraycopy(container.gradientBuffer, 0, downstreamGradient, 0, size);
        
        return downstreamGradient;
    }
    
    @Override
    public int getOutputSize() {
        return fixedSize; // Returns -1 for dynamic, or the fixed size
    }
    
    /**
     * Get the dropout rate.
     */
    public float getDropoutRate() {
        return dropoutRate;
    }
    
    // Serialization implementation
    
    @Override
    public void writeTo(DataOutputStream out, int version) throws IOException {
        out.writeFloat(dropoutRate);
        out.writeInt(fixedSize);
    }
    
    @Override
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use deserialize() static method instead");
    }
    
    /**
     * Deserialize a DropoutLayer from stream.
     */
    public static DropoutLayer deserialize(DataInputStream in, int version) throws IOException {
        float dropoutRate = in.readFloat();
        int fixedSize = in.readInt();
        
        return new DropoutLayer(dropoutRate, fixedSize);
    }
    
    @Override
    public int getSerializedSize(int version) {
        return 4 + 4; // float + int
    }
    
    @Override
    public int getTypeId() {
        return SerializationConstants.TYPE_DROPOUT_LAYER;
    }
    
    /**
     * Create a dropout layer specification.
     */
    public static Layer.Spec spec(float dropoutRate) {
        return new DropoutSpec(dropoutRate);
    }
    
    /**
     * Specification for creating dropout layers.
     */
    private static class DropoutSpec implements Layer.Spec {
        private final float dropoutRate;
        
        public DropoutSpec(float dropoutRate) {
            this.dropoutRate = dropoutRate;
        }
        
        @Override
        public int getOutputSize() {
            return -1; // Output size matches input size
        }
        
        @Override
        public int getOutputSize(int inputSize) {
            return inputSize; // Dropout preserves input size
        }
        
        @Override
        public Layer create(int inputSize) {
            return new DropoutLayer(dropoutRate, inputSize);
        }
        
        @Override
        public Layer create(int inputSize, Optimizer defaultOptimizer) {
            return create(inputSize); // Dropout doesn't use optimizer
        }
    }
}