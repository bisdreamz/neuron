package dev.neuronic.net.layers;

import dev.neuronic.net.common.PooledFloatArray;
import dev.neuronic.net.WeightInitStrategy;
import dev.neuronic.net.activators.*;
import dev.neuronic.net.math.NetMath;
import dev.neuronic.net.losses.CombinedLossActivation;
import dev.neuronic.net.losses.Loss;
import dev.neuronic.net.optimizers.Optimizer;
import dev.neuronic.net.optimizers.AdamOptimizer;
import dev.neuronic.net.optimizers.AdamWOptimizer;
import dev.neuronic.net.optimizers.SgdOptimizer;
import dev.neuronic.net.serialization.Serializable;
import dev.neuronic.net.serialization.SerializationConstants;
import dev.neuronic.net.serialization.SerializationRegistry;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;

public class DenseLayer implements Layer, GradientAccumulator, Serializable {

    protected final Optimizer optimizer;
    protected final Activator activator;
    protected final float[][] weights; // Column-major: weights[input][neuron] for better vectorization
    protected final float[] biases;
    protected final int neurons;
    protected final int inputs;
    protected final ThreadLocal<float[]> preActivationBuffers;
    protected final ThreadLocal<float[]> activationOutputBuffers;
    protected final ThreadLocal<float[]> activationDerivativeBuffers;
    protected final ThreadLocal<float[]> neuronDeltaBuffers;
    protected final ThreadLocal<float[]> inputBuffers;
    protected final ThreadLocal<float[][]> weightGradientBuffers;
    
    // Gradient accumulation state
    protected final ThreadLocal<float[][]> accumulatedWeightGradients;
    protected final ThreadLocal<float[]> accumulatedBiasGradients;
    protected final ThreadLocal<Boolean> accumulating;
    
    // Buffer pools for temporary gradient arrays - shared across all DenseLayer instances
    private static final ConcurrentHashMap<Integer, PooledFloatArray> biasGradientPools = new ConcurrentHashMap<>();
    private static final ConcurrentHashMap<Integer, PooledFloatArray> weightGradientRowPools = new ConcurrentHashMap<>();

    public DenseLayer(Optimizer optimizer, Activator activator, int neurons, int inputs, WeightInitStrategy initStrategy) {
        this.optimizer = optimizer;
        this.activator = activator;
        this.weights = new float[inputs][neurons]; // Column-major layout
        this.biases = new float[neurons];
        this.neurons = neurons;
        this.inputs = inputs;
        this.preActivationBuffers = ThreadLocal.withInitial(() -> new float[neurons]);
        this.activationOutputBuffers = ThreadLocal.withInitial(() -> new float[neurons]);
        this.activationDerivativeBuffers = ThreadLocal.withInitial(() -> new float[neurons]);
        this.neuronDeltaBuffers = ThreadLocal.withInitial(() -> new float[neurons]);
        this.inputBuffers = ThreadLocal.withInitial(() -> new float[inputs]);
        this.weightGradientBuffers = ThreadLocal.withInitial(() -> new float[inputs][neurons]);
        
        // Initialize gradient accumulation state
        this.accumulatedWeightGradients = ThreadLocal.withInitial(() -> new float[inputs][neurons]);
        this.accumulatedBiasGradients = ThreadLocal.withInitial(() -> new float[neurons]);
        this.accumulating = ThreadLocal.withInitial(() -> Boolean.FALSE);

        // Initialize weights and biases based on strategy
        switch (initStrategy) {
            case XAVIER -> NetMath.weightInitXavier(weights, inputs, neurons);
            case HE -> NetMath.weightInitHe(weights, inputs);
        }
        NetMath.biasInit(biases, 0.0f);
    }

    public Layer.LayerContext forward(float[] input) {
        // Allocate new arrays for LayerContext - never use ThreadLocal buffers in contexts
        float[] preActivations = new float[neurons];
        NetMath.matrixPreActivationsColumnMajor(input, weights, biases, preActivations);
        
        float[] activationOutputs = new float[neurons];
        this.activator.activate(preActivations, activationOutputs);
        
        return new Layer.LayerContext(input, preActivations, activationOutputs);
    }

    public float[] backward(Layer.LayerContext[] stack, int stackIndex, float[] upstreamGradient) {
        return backward(stack, stackIndex, upstreamGradient, (Loss) null);
    }
    
    public float[] backward(Layer.LayerContext[] stack, int stackIndex, float[] upstreamGradient, 
                          Loss loss) {
        Layer.LayerContext context = stack[stackIndex];
        float[] neuronDeltas;
        
        // Check if this is the final layer and loss handles activation derivatives
        boolean isFinalLayer = stackIndex == stack.length - 1;
        boolean skipActivationDerivatives = false;
        
        if (isFinalLayer && loss instanceof CombinedLossActivation) {
            CombinedLossActivation combined =
                (CombinedLossActivation) loss;
            skipActivationDerivatives = combined.getHandledActivator().equals(activator.getClass());
        }

        if (skipActivationDerivatives) {
            // Loss function already computed combined derivative - use upstream gradient directly
            neuronDeltas = upstreamGradient.clone();
        } else {
            // Standard backpropagation: apply activation derivatives
            float[] activationDerivatives = activationDerivativeBuffers.get();
            activator.derivative(context.preActivations(), activationDerivatives);

            neuronDeltas = neuronDeltaBuffers.get();
            NetMath.elementwiseMultiply(activationDerivatives, upstreamGradient, neuronDeltas);
        }

        // Compute weight gradients using optimized column-major computation
        float[][] weightGradients = weightGradientBuffers.get();
        NetMath.matrixWeightGradientsColumnMajor(context.inputs(), neuronDeltas, weightGradients);

        // Update weights and biases
        optimizer.optimize(weights, biases, weightGradients, neuronDeltas);

        // Compute downstream gradient using optimized matrix-vector multiplication
        float[] downstreamGradient = inputBuffers.get();
        NetMath.matrixVectorMultiplyColumnMajor(weights, neuronDeltas, downstreamGradient);
        
        return downstreamGradient;
    }
    
    @Override
    public LayerContext forward(float[] input, ExecutorService executor) {
        // Allocate new arrays for LayerContext - never use ThreadLocal buffers in contexts
        float[] preActivations = new float[neurons];
        NetMath.matrixPreActivationsColumnMajor(input, weights, biases, preActivations);
        
        float[] activationOutputs = new float[neurons];
        // Use executor-aware activation
        this.activator.activate(preActivations, activationOutputs, executor);
        
        return new Layer.LayerContext(input, preActivations, activationOutputs);
    }
    
    @Override
    public float[] backward(LayerContext[] stack, int stackIndex, float[] upstreamGradient, ExecutorService executor) {
        return backward(stack, stackIndex, upstreamGradient, null, executor);
    }
    
    public float[] backward(Layer.LayerContext[] stack, int stackIndex, float[] upstreamGradient,
                            Loss loss, ExecutorService executor) {
        Layer.LayerContext context = stack[stackIndex];
        float[] neuronDeltas;
        
        // Check if this is the final layer and loss handles activation derivatives
        boolean isFinalLayer = stackIndex == stack.length - 1;
        boolean skipActivationDerivatives = false;
        
        if (isFinalLayer && loss instanceof CombinedLossActivation) {
            CombinedLossActivation combined =
                (CombinedLossActivation) loss;
            skipActivationDerivatives = combined.getHandledActivator().equals(activator.getClass());
        }

        if (skipActivationDerivatives) {
            // Loss function already computed combined derivative - use upstream gradient directly
            neuronDeltas = upstreamGradient.clone();
        } else {
            // Standard backpropagation: apply activation derivatives
            float[] activationDerivatives = activationDerivativeBuffers.get();
            activator.derivative(context.preActivations(), activationDerivatives, executor);

            neuronDeltas = neuronDeltaBuffers.get();
            NetMath.elementwiseMultiply(activationDerivatives, upstreamGradient, neuronDeltas);
        }

        // Compute weight gradients using optimized column-major computation
        float[][] weightGradients = weightGradientBuffers.get();
        NetMath.matrixWeightGradientsColumnMajor(context.inputs(), neuronDeltas, weightGradients);

        // Update weights and biases with executor
        optimizer.optimize(weights, biases, weightGradients, neuronDeltas, executor);

        // Compute downstream gradient using optimized matrix-vector multiplication
        float[] downstreamGradient = inputBuffers.get();
        NetMath.matrixVectorMultiplyColumnMajor(weights, neuronDeltas, downstreamGradient);
        
        return downstreamGradient;
    }
    
    @Override
    public float[] computeGradient(LayerContext[] stack, int stackIndex, 
                                  float[] upstreamGradient, GradientConsumer gradientConsumer) {
        LayerContext context = stack[stackIndex];
        
        // Apply activation derivative
        float[] activationDerivatives = activationDerivativeBuffers.get();
        activator.derivative(context.preActivations(), activationDerivatives);
        
        float[] neuronDeltas = neuronDeltaBuffers.get();
        NetMath.elementwiseMultiply(activationDerivatives, upstreamGradient, neuronDeltas);
        
        // Compute weight gradients
        float[][] weightGradients = weightGradientBuffers.get();
        NetMath.matrixWeightGradientsColumnMajor(context.inputs(), neuronDeltas, weightGradients);
        
        // Pass gradients to consumer if provided
        if (gradientConsumer != null) {
            gradientConsumer.accept(stackIndex, weightGradients, neuronDeltas);
        }
        
        // Compute downstream gradient
        float[] downstreamGradient = inputBuffers.get();
        NetMath.matrixVectorMultiplyColumnMajor(weights, neuronDeltas, downstreamGradient);
        
        return downstreamGradient;
    }
    
    @Override
    public void applyGradients(float[][] weightGradients, float[] biasGradients) {
        optimizer.optimize(weights, biases, weightGradients, biasGradients);
    }
    
    @Override
    public GradientDimensions getGradientDimensions() {
        return new GradientDimensions(inputs, neurons, neurons);
    }
    
    @Override
    public int getOutputSize() {
        return neurons;
    }
    
    @Override
    public Optimizer getOptimizer() {
        return optimizer;
    }
    
    // Gradient accumulation implementation
    
    @Override
    public void startAccumulation() {
        accumulating.set(Boolean.TRUE);
        // Zero out accumulated gradients
        float[][] weightGrads = accumulatedWeightGradients.get();
        float[] biasGrads = accumulatedBiasGradients.get();
        
        // Zero out using vectorized operations
        NetMath.matrixInit(weightGrads, 0.0f);
        NetMath.biasInit(biasGrads, 0.0f);
    }
    
    @Override
    public float[] backwardAccumulate(Layer.LayerContext[] stack, int stackIndex, float[] upstreamGradient) {
        Layer.LayerContext context = stack[stackIndex];
        float[] neuronDeltas;
        
        // Check if this is the final layer and loss handles activation derivatives
        boolean isFinalLayer = stackIndex == stack.length - 1;
        
        if (isFinalLayer && activator instanceof SoftmaxActivator) {
            // Assume combined loss-activation optimization
            neuronDeltas = upstreamGradient.clone();
        } else {
            // Standard backpropagation: apply activation derivatives
            float[] activationDerivatives = activationDerivativeBuffers.get();
            activator.derivative(context.preActivations(), activationDerivatives);

            neuronDeltas = neuronDeltaBuffers.get();
            NetMath.elementwiseMultiply(activationDerivatives, upstreamGradient, neuronDeltas);
        }

        // Compute weight gradients
        float[][] weightGradients = weightGradientBuffers.get();
        NetMath.matrixWeightGradientsColumnMajor(context.inputs(), neuronDeltas, weightGradients);

        // Accumulate gradients using NetMath operations
        float[][] accWeightGrads = accumulatedWeightGradients.get();
        float[] accBiasGrads = accumulatedBiasGradients.get();
        
        // Add weight gradients to accumulated - process row by row
        for (int i = 0; i < inputs; i++) {
            NetMath.elementwiseAdd(accWeightGrads[i], weightGradients[i], accWeightGrads[i]);
        }
        
        // Add bias gradients to accumulated
        NetMath.elementwiseAdd(accBiasGrads, neuronDeltas, accBiasGrads);

        // Compute downstream gradient
        float[] downstreamGradient = inputBuffers.get();
        NetMath.matrixVectorMultiplyColumnMajor(weights, neuronDeltas, downstreamGradient);
        
        return downstreamGradient;
    }
    
    @Override
    public void applyAccumulatedGradients(int batchSize) {
        if (!accumulating.get()) return;
        
        float[][] accWeightGrads = accumulatedWeightGradients.get();
        float[] accBiasGrads = accumulatedBiasGradients.get();
        
        // Get temporary buffers from pools
        PooledFloatArray biasPool = getBiasGradientPool();
        float[] tempBiasGrads = biasPool.getBuffer(false); // false = no need to zero, we'll overwrite
        
        float[][] tempWeightGrads = new float[inputs][];
        PooledFloatArray rowPool = getWeightGradientRowPool();
        for (int i = 0; i < inputs; i++) {
            tempWeightGrads[i] = rowPool.getBuffer(false);
        }
        
        try {
            // Copy and scale gradients into temporary buffers
            float scale = 1.0f / batchSize;
            
            // Copy and scale weight gradients
            for (int i = 0; i < inputs; i++) {
                System.arraycopy(accWeightGrads[i], 0, tempWeightGrads[i], 0, neurons);
                NetMath.elementwiseScaleInPlace(tempWeightGrads[i], scale);
            }
            
            // Copy and scale bias gradients
            System.arraycopy(accBiasGrads, 0, tempBiasGrads, 0, neurons);
            NetMath.elementwiseScaleInPlace(tempBiasGrads, scale);
            
            // Update weights and biases using temporary scaled gradients
            optimizer.optimize(weights, biases, tempWeightGrads, tempBiasGrads);
            
        } finally {
            // Always return buffers to pools
            biasPool.releaseBuffer(tempBiasGrads);
            for (int i = 0; i < inputs; i++) {
                rowPool.releaseBuffer(tempWeightGrads[i]);
            }
            
            accumulating.set(Boolean.FALSE);
        }
    }
    
    @Override
    public boolean isAccumulating() {
        return accumulating.get();
    }
    
    /**
     * Package-private method to compute the L2 norm squared of accumulated gradients.
     * Used by NeuralNet for global gradient clipping.
     */
    public float getAccumulatedGradientNormSquared() {
        if (!accumulating.get()) return 0.0f;
        
        float normSquared = 0.0f;
        float[][] weightGrads = accumulatedWeightGradients.get();
        float[] biasGrads = accumulatedBiasGradients.get();
        
        // Sum squares of weight gradients
        for (int i = 0; i < inputs; i++) {
            for (int j = 0; j < neurons; j++) {
                float grad = weightGrads[i][j];
                normSquared += grad * grad;
            }
        }
        
        // Sum squares of bias gradients
        for (int j = 0; j < neurons; j++) {
            float grad = biasGrads[j];
            normSquared += grad * grad;
        }
        
        return normSquared;
    }
    
    /**
     * Package-private method to apply accumulated gradients with optional scaling.
     * Used by NeuralNet for global gradient clipping.
     */
    public void applyAccumulatedGradients(int batchSize, float scaleFactor) {
        if (!accumulating.get()) return;
        
        float[][] accWeightGrads = accumulatedWeightGradients.get();
        float[] accBiasGrads = accumulatedBiasGradients.get();
        
        // Get temporary buffers from pools
        PooledFloatArray biasPool = getBiasGradientPool();
        float[] tempBiasGrads = biasPool.getBuffer(false);
        
        float[][] tempWeightGrads = new float[inputs][];
        PooledFloatArray rowPool = getWeightGradientRowPool();
        for (int i = 0; i < inputs; i++) {
            tempWeightGrads[i] = rowPool.getBuffer(false);
        }
        
        try {
            // Copy and scale gradients into temporary buffers
            float scale = scaleFactor / batchSize;
            
            // Copy and scale weight gradients
            for (int i = 0; i < inputs; i++) {
                System.arraycopy(accWeightGrads[i], 0, tempWeightGrads[i], 0, neurons);
                NetMath.elementwiseScaleInPlace(tempWeightGrads[i], scale);
            }
            
            // Copy and scale bias gradients
            System.arraycopy(accBiasGrads, 0, tempBiasGrads, 0, neurons);
            NetMath.elementwiseScaleInPlace(tempBiasGrads, scale);
            
            // Update weights and biases using temporary scaled gradients
            optimizer.optimize(weights, biases, tempWeightGrads, tempBiasGrads);
            
        } finally {
            // Always return buffers to pools
            biasPool.releaseBuffer(tempBiasGrads);
            for (int i = 0; i < inputs; i++) {
                rowPool.releaseBuffer(tempWeightGrads[i]);
            }
            
            accumulating.set(Boolean.FALSE);
        }
    }
    
    /**
     * Create a layer specification for a dense layer.
     * Weight initialization strategy must be specified explicitly to prevent silent bugs.
     * 
     * @param neurons number of neurons in this layer
     * @param activator activation function (choose init strategy accordingly)
     * @param optimizer optimizer for this layer
     * @param initStrategy weight initialization strategy:
     *                     - HE: for ReLU and variants
     *                     - XAVIER: for Sigmoid, Tanh, Softmax
     */
    public static Layer.Spec spec(int neurons, Activator activator, Optimizer optimizer, WeightInitStrategy initStrategy) {
        return new DenseLayerSpec(neurons, activator, optimizer, initStrategy, 1.0);
    }
    
    /**
     * Create a dense layer specification with optional optimizer (for use with default optimizer).
     * Returns a chainable spec that allows setting optimizer and learning rate ratio.
     * 
     * @param neurons number of neurons in this layer
     * @param activator activation function
     * @param initStrategy weight initialization strategy
     * @return chainable layer specification
     */
    public static DenseLayerSpec specChainable(int neurons, Activator activator, WeightInitStrategy initStrategy) {
        return new DenseLayerSpec(neurons, activator, null, initStrategy, 1.0);
    }
    
    /**
     * Create a layer specification with custom learning rate ratio.
     * 
     * @param neurons number of neurons in this layer
     * @param activator activation function
     * @param optimizer optimizer for this layer (null to use default)
     * @param initStrategy weight initialization strategy
     * @param learningRateRatio learning rate scaling factor (1.0 = normal)
     */
    public static Layer.Spec spec(int neurons, Activator activator, Optimizer optimizer, 
                                  WeightInitStrategy initStrategy, double learningRateRatio) {
        return new DenseLayerSpec(neurons, activator, optimizer, initStrategy, learningRateRatio);
    }
    
    /**
     * Specification for creating dense layers with optimizer management.
     */
    public static class DenseLayerSpec extends BaseLayerSpec<DenseLayerSpec> {
        private final int neurons;
        private final Activator activator;
        private final WeightInitStrategy initStrategy;
        
        public DenseLayerSpec(int neurons, Activator activator, Optimizer optimizer, 
                              WeightInitStrategy initStrategy, double learningRateRatio) {
            super(neurons, optimizer);
            this.neurons = neurons;
            this.activator = activator;
            this.initStrategy = initStrategy;
            this.learningRateRatio = (float) learningRateRatio;
        }
        
        @Override
        public Layer create(int inputSize) {
            return createLayer(inputSize, getEffectiveOptimizer(null));
        }
        
        @Override
        protected Layer createLayer(int inputSize, Optimizer effectiveOptimizer) {
            return new DenseLayer(effectiveOptimizer, activator, neurons, inputSize, initStrategy);
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
        
        // Write activator (built-in type ID or custom class name)
        writeActivator(out, activator);
        
        // Write optimizer (built-in type ID or custom class name)
        writeOptimizer(out, optimizer, version);
    }
    
    @Override
    public void readFrom(DataInputStream in, int version) throws IOException {
        throw new UnsupportedOperationException("Use readFrom(DataInputStream, int) static method instead");
    }
    
    /**
     * Static method to deserialize a DenseLayer from stream.
     */
    public static DenseLayer deserialize(DataInputStream in, int version) throws IOException {
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
        
        // Read activator
        Activator activator = readActivator(in);
        
        // Read optimizer
        Optimizer optimizer = readOptimizer(in, version);
        
        // Create layer and set deserialized values
        DenseLayer layer = new DenseLayer(optimizer, activator, neurons, inputs, WeightInitStrategy.XAVIER);
        
        // Copy weights and biases to the new layer
        System.arraycopy(biases, 0, layer.biases, 0, neurons);
        for (int i = 0; i < inputs; i++) {
            System.arraycopy(weights[i], 0, layer.weights[i], 0, neurons);
        }
        
        return layer;
    }
    
    private static void writeActivator(DataOutputStream out, Activator activator) throws IOException {
        // Check if it's a registered custom activator
        String registeredName = SerializationRegistry.getRegisteredName(activator);
        if (registeredName != null) {
            out.writeInt(SerializationConstants.TYPE_CUSTOM);
            out.writeUTF(registeredName);
            return;
        }
        
        // Use built-in type ID
        String className = activator.getClass().getSimpleName();
        int typeId = switch (className) {
            case "ReluActivator" -> SerializationConstants.TYPE_RELU_ACTIVATOR;
            case "SigmoidActivator" -> SerializationConstants.TYPE_SIGMOID_ACTIVATOR;
            case "TanhActivator" -> SerializationConstants.TYPE_TANH_ACTIVATOR;
            case "SoftmaxActivator" -> SerializationConstants.TYPE_SOFTMAX_ACTIVATOR;
            case "LinearActivator" -> SerializationConstants.TYPE_LINEAR_ACTIVATOR;
            case "LeakyReluActivator" -> SerializationConstants.TYPE_LEAKY_RELU_ACTIVATOR;
            default -> throw new IllegalArgumentException("Unknown activator type: " + className + 
                " (register custom activators with SerializationRegistry)");
        };
        out.writeInt(typeId);
        
        // Write additional state for stateful activators
        if (activator instanceof LeakyReluActivator)
            out.writeFloat(((LeakyReluActivator) activator).getAlpha());
    }
    
    private static Activator readActivator(DataInputStream in) throws IOException {
        int typeId = in.readInt();
        
        if (typeId == SerializationConstants.TYPE_CUSTOM) {
            String className = in.readUTF();
            return SerializationRegistry.createActivator(className);
        }
        
        return switch (typeId) {
            case SerializationConstants.TYPE_RELU_ACTIVATOR -> ReluActivator.INSTANCE;
            case SerializationConstants.TYPE_SIGMOID_ACTIVATOR -> SigmoidActivator.INSTANCE;
            case SerializationConstants.TYPE_TANH_ACTIVATOR -> TanhActivator.INSTANCE;
            case SerializationConstants.TYPE_SOFTMAX_ACTIVATOR -> SoftmaxActivator.INSTANCE;
            case SerializationConstants.TYPE_LINEAR_ACTIVATOR -> LinearActivator.INSTANCE;
            case SerializationConstants.TYPE_LEAKY_RELU_ACTIVATOR -> {
                float alpha = in.readFloat();
                yield LeakyReluActivator.create(alpha);
            }
            default -> throw new IllegalArgumentException("Unknown activator type ID: " + typeId);
        };
    }
    
    private static void writeOptimizer(DataOutputStream out, Optimizer optimizer, int version) throws IOException {
        // Check if it's a registered custom optimizer
        String registeredName = SerializationRegistry.getRegisteredName(optimizer);
        if (registeredName != null) {
            out.writeInt(SerializationConstants.TYPE_CUSTOM);
            out.writeUTF(registeredName);
            // Custom optimizers handle their own serialization
            return;
        }
        
        // Use built-in serialization
        Serializable serializableOptimizer = (Serializable) optimizer;
        out.writeInt(serializableOptimizer.getTypeId());
        serializableOptimizer.writeTo(out, version);
    }
    
    private static Optimizer readOptimizer(DataInputStream in, int version) throws IOException {
        int typeId = in.readInt();
        
        if (typeId == SerializationConstants.TYPE_CUSTOM) {
            String className = in.readUTF();
            return SerializationRegistry.createOptimizer(className, in, version);
        }
        
        return switch (typeId) {
            case SerializationConstants.TYPE_SGD_OPTIMIZER -> SgdOptimizer.deserialize(in, version);
            case SerializationConstants.TYPE_ADAM_OPTIMIZER -> AdamOptimizer.deserialize(in, version);
            case SerializationConstants.TYPE_ADAMW_OPTIMIZER -> AdamWOptimizer.deserialize(in, version);
            default -> throw new IOException("Unknown optimizer type ID: " + typeId);
        };
    }
    
    @Override
    public int getSerializedSize(int version) {
        int size = 8; // neurons + inputs
        size += inputs * neurons * 4; // weights
        size += neurons * 4; // biases
        
        // Activator size (type ID + possible class name)
        String registeredActivatorName = SerializationRegistry.getRegisteredName(activator);
        if (registeredActivatorName != null) {
            size += 4; // TYPE_CUSTOM
            size += 2 + registeredActivatorName.getBytes().length; // UTF string
        } else {
            size += 4; // built-in type ID
        }
        
        // Optimizer size (type ID + data + possible class name)
        String registeredOptimizerName = SerializationRegistry.getRegisteredName(optimizer);
        if (registeredOptimizerName != null) {
            size += 4; // TYPE_CUSTOM
            size += 2 + registeredOptimizerName.getBytes().length; // UTF string
            // Custom optimizers handle their own serialization via factory
        } else {
            size += 4; // built-in type ID
            size += ((Serializable) optimizer).getSerializedSize(version); // optimizer data
        }
        
        return size;
    }
    
    @Override
    public int getTypeId() {
        return SerializationConstants.TYPE_DENSE_LAYER;
    }
    
    /**
     * Get or create a buffer pool for bias gradients of this layer's size.
     */
    private PooledFloatArray getBiasGradientPool() {
        return biasGradientPools.computeIfAbsent(neurons, PooledFloatArray::new);
    }
    
    /**
     * Get or create a buffer pool for weight gradient rows of this layer's size.
     */
    private PooledFloatArray getWeightGradientRowPool() {
        return weightGradientRowPools.computeIfAbsent(neurons, PooledFloatArray::new);
    }

}
