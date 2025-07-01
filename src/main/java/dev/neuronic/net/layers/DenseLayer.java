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
    // Instance buffer pools - one pool per buffer size
    private final PooledFloatArray neuronBufferPool;      // For neuron-sized arrays
    private final PooledFloatArray inputBufferPool;       // For input-sized arrays
    
    // Gradient accumulation state - these need to persist between calls
    protected final ThreadLocal<float[][]> accumulatedWeightGradients;
    protected final ThreadLocal<float[]> accumulatedBiasGradients;
    protected final ThreadLocal<Boolean> accumulating;

    public DenseLayer(Optimizer optimizer, Activator activator, int neurons, int inputs, WeightInitStrategy initStrategy) {
        this.optimizer = optimizer;
        this.activator = activator;
        this.weights = new float[inputs][neurons]; // Column-major layout
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

        // Initialize weights and biases based on strategy
        switch (initStrategy) {
            case XAVIER -> NetMath.weightInitXavier(weights, inputs, neurons);
            case HE -> NetMath.weightInitHe(weights, inputs);
        }
        NetMath.biasInit(biases, 0.0f);
    }

    public Layer.LayerContext forward(float[] input, boolean isTraining) {
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
        
        // Check if this is the final layer and loss handles activation derivatives
        boolean isFinalLayer = stackIndex == stack.length - 1;
        boolean skipActivationDerivatives = false;
        
        if (isFinalLayer && loss instanceof CombinedLossActivation) {
            CombinedLossActivation combined =
                (CombinedLossActivation) loss;
            skipActivationDerivatives = combined.getHandledActivator().equals(activator.getClass());
        }

        float[] neuronDeltas = null;
        float[] activationDerivatives = null;
        float[] downstreamGradient = null;
        
        try {
            if (skipActivationDerivatives) {
                neuronDeltas = upstreamGradient.clone();
            } else {
                activationDerivatives = neuronBufferPool.getBuffer();
                activator.derivative(context.preActivations(), activationDerivatives);

                neuronDeltas = neuronBufferPool.getBuffer();
                NetMath.elementwiseMultiply(activationDerivatives, upstreamGradient, neuronDeltas);
            }

            float[][] weightGradients = new float[inputs][neurons];
            NetMath.matrixWeightGradientsColumnMajor(context.inputs(), neuronDeltas, weightGradients);

            optimizer.optimize(weights, biases, weightGradients, neuronDeltas);

            downstreamGradient = inputBufferPool.getBuffer();
            NetMath.matrixVectorMultiplyColumnMajor(weights, neuronDeltas, downstreamGradient);
            
            float[] result = new float[inputs];
            System.arraycopy(downstreamGradient, 0, result, 0, inputs);
            return result;
            
        } finally {
            if (activationDerivatives != null) {
                neuronBufferPool.releaseBuffer(activationDerivatives);
            }
            // Don't release neuronDeltas if it's a clone of upstreamGradient
            if (neuronDeltas != null && !skipActivationDerivatives) {
                neuronBufferPool.releaseBuffer(neuronDeltas);
            }
            if (downstreamGradient != null) {
                inputBufferPool.releaseBuffer(downstreamGradient);
            }
        }
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
        
        boolean isFinalLayer = stackIndex == stack.length - 1;
        boolean skipActivationDerivatives = false;
        
        if (isFinalLayer && loss instanceof CombinedLossActivation) {
            CombinedLossActivation combined =
                (CombinedLossActivation) loss;
            skipActivationDerivatives = combined.getHandledActivator().equals(activator.getClass());
        }

        float[] neuronDeltas = null;
        float[] activationDerivatives = null;
        float[] downstreamGradient = null;
        
        try {
            if (skipActivationDerivatives) {
                neuronDeltas = upstreamGradient.clone();
            } else {
                activationDerivatives = neuronBufferPool.getBuffer();
                activator.derivative(context.preActivations(), activationDerivatives, executor);

                neuronDeltas = neuronBufferPool.getBuffer();
                NetMath.elementwiseMultiply(activationDerivatives, upstreamGradient, neuronDeltas);
            }

            float[][] weightGradients = new float[inputs][neurons];
            NetMath.matrixWeightGradientsColumnMajor(context.inputs(), neuronDeltas, weightGradients);

            optimizer.optimize(weights, biases, weightGradients, neuronDeltas, executor);

            downstreamGradient = inputBufferPool.getBuffer();
            NetMath.matrixVectorMultiplyColumnMajor(weights, neuronDeltas, downstreamGradient);
            
            float[] result = new float[inputs];
            System.arraycopy(downstreamGradient, 0, result, 0, inputs);
            return result;
            
        } finally {
            if (activationDerivatives != null) {
                neuronBufferPool.releaseBuffer(activationDerivatives);
            }
            // Don't release neuronDeltas if it's a clone of upstreamGradient
            if (neuronDeltas != null && !skipActivationDerivatives) {
                neuronBufferPool.releaseBuffer(neuronDeltas);
            }
            if (downstreamGradient != null) {
                inputBufferPool.releaseBuffer(downstreamGradient);
            }
        }
    }
    
    @Override
    public float[] computeGradient(LayerContext[] stack, int stackIndex, 
                                  float[] upstreamGradient, GradientConsumer gradientConsumer) {
        LayerContext context = stack[stackIndex];
        
        float[] activationDerivatives = null;
        float[] neuronDeltas = null;
        float[] downstreamGradient = null;
        
        try {
            activationDerivatives = neuronBufferPool.getBuffer();
            activator.derivative(context.preActivations(), activationDerivatives);
            
            neuronDeltas = neuronBufferPool.getBuffer();
            NetMath.elementwiseMultiply(activationDerivatives, upstreamGradient, neuronDeltas);
            
            float[][] weightGradients = new float[inputs][neurons];
            NetMath.matrixWeightGradientsColumnMajor(context.inputs(), neuronDeltas, weightGradients);
            
            if (gradientConsumer != null) {
                gradientConsumer.accept(stackIndex, weightGradients, neuronDeltas);
            }
            
            downstreamGradient = inputBufferPool.getBuffer();
            NetMath.matrixVectorMultiplyColumnMajor(weights, neuronDeltas, downstreamGradient);
            
            float[] result = new float[inputs];
            System.arraycopy(downstreamGradient, 0, result, 0, inputs);
            return result;
            
        } finally {
            if (activationDerivatives != null) {
                neuronBufferPool.releaseBuffer(activationDerivatives);
            }
            if (neuronDeltas != null) {
                neuronBufferPool.releaseBuffer(neuronDeltas);
            }
            if (downstreamGradient != null) {
                inputBufferPool.releaseBuffer(downstreamGradient);
            }
        }
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
        
        boolean isFinalLayer = stackIndex == stack.length - 1;
        
        float[] neuronDeltas = null;
        float[] activationDerivatives = null;
        float[] downstreamGradient = null;
        
        try {
            if (isFinalLayer && activator instanceof SoftmaxActivator) {
                neuronDeltas = upstreamGradient.clone();
            } else {
                activationDerivatives = neuronBufferPool.getBuffer();
                activator.derivative(context.preActivations(), activationDerivatives);

                neuronDeltas = neuronBufferPool.getBuffer();
                NetMath.elementwiseMultiply(activationDerivatives, upstreamGradient, neuronDeltas);
            }

            float[][] weightGradients = new float[inputs][neurons];
            NetMath.matrixWeightGradientsColumnMajor(context.inputs(), neuronDeltas, weightGradients);

            float[][] accWeightGrads = accumulatedWeightGradients.get();
            float[] accBiasGrads = accumulatedBiasGradients.get();
        
            // Add weight gradients to accumulated - process row by row
            for (int i = 0; i < inputs; i++) {
                NetMath.elementwiseAdd(accWeightGrads[i], weightGradients[i], accWeightGrads[i]);
            }
            
            // Add bias gradients to accumulated
            NetMath.elementwiseAdd(accBiasGrads, neuronDeltas, accBiasGrads);

            // Compute downstream gradient
            downstreamGradient = inputBufferPool.getBuffer();
            NetMath.matrixVectorMultiplyColumnMajor(weights, neuronDeltas, downstreamGradient);
            
            float[] result = new float[inputs];
            System.arraycopy(downstreamGradient, 0, result, 0, inputs);
            return result;
            
        } finally {
            if (activationDerivatives != null && !(isFinalLayer && activator instanceof SoftmaxActivator)) {
                neuronBufferPool.releaseBuffer(activationDerivatives);
            }
            if (neuronDeltas != null && !(isFinalLayer && activator instanceof SoftmaxActivator)) {
                neuronBufferPool.releaseBuffer(neuronDeltas);
            }
            if (downstreamGradient != null) {
                inputBufferPool.releaseBuffer(downstreamGradient);
            }
        }
    }
    
    @Override
    public void applyAccumulatedGradients(int batchSize) {
        if (!accumulating.get()) return;
        
        float[][] accWeightGrads = accumulatedWeightGradients.get();
        float[] accBiasGrads = accumulatedBiasGradients.get();
        
        float[] tempBiasGrads = neuronBufferPool.getBuffer(false);
        
        float[][] tempWeightGrads = new float[inputs][];
        for (int i = 0; i < inputs; i++) {
            tempWeightGrads[i] = neuronBufferPool.getBuffer(false);
        }
        
        try {
            float scale = 1.0f / batchSize;
            
            for (int i = 0; i < inputs; i++) {
                System.arraycopy(accWeightGrads[i], 0, tempWeightGrads[i], 0, neurons);
                NetMath.elementwiseScaleInPlace(tempWeightGrads[i], scale);
            }
            
            System.arraycopy(accBiasGrads, 0, tempBiasGrads, 0, neurons);
            NetMath.elementwiseScaleInPlace(tempBiasGrads, scale);
            
            optimizer.optimize(weights, biases, tempWeightGrads, tempBiasGrads);
            
        } finally {
            neuronBufferPool.releaseBuffer(tempBiasGrads);
            for (int i = 0; i < inputs; i++) {
                neuronBufferPool.releaseBuffer(tempWeightGrads[i]);
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
        
        float[] tempBiasGrads = neuronBufferPool.getBuffer(false);
        
        float[][] tempWeightGrads = new float[inputs][];
        for (int i = 0; i < inputs; i++) {
            tempWeightGrads[i] = neuronBufferPool.getBuffer(false);
        }
        
        try {
            float scale = scaleFactor / batchSize;
            
            for (int i = 0; i < inputs; i++) {
                System.arraycopy(accWeightGrads[i], 0, tempWeightGrads[i], 0, neurons);
                NetMath.elementwiseScaleInPlace(tempWeightGrads[i], scale);
            }
            
            System.arraycopy(accBiasGrads, 0, tempBiasGrads, 0, neurons);
            NetMath.elementwiseScaleInPlace(tempBiasGrads, scale);
            
            optimizer.optimize(weights, biases, tempWeightGrads, tempBiasGrads);
            
        } finally {
            neuronBufferPool.releaseBuffer(tempBiasGrads);
            for (int i = 0; i < inputs; i++) {
                neuronBufferPool.releaseBuffer(tempWeightGrads[i]);
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
     * Get the weights array for testing purposes.
     * Returns a flattened view of the weights in row-major order.
     */
    public float[] getWeights() {
        float[] flattened = new float[inputs * neurons];
        int idx = 0;
        for (int i = 0; i < inputs; i++) {
            for (int j = 0; j < neurons; j++) {
                flattened[idx++] = weights[i][j];
            }
        }
        return flattened;
    }
    
    /**
     * Set a specific weight for testing purposes.
     * @param flatIndex the flattened index
     * @param value the weight value
     */
    public void setWeight(int flatIndex, float value) {
        int i = flatIndex / neurons;
        int j = flatIndex % neurons;
        weights[i][j] = value;
    }
    
    /**
     * Set a specific bias for testing purposes.
     * @param index the bias index
     * @param value the bias value
     */
    public void setBias(int index, float value) {
        biases[index] = value;
    }

}
